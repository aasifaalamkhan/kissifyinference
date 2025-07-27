# inference_server.py - With authentication and refactored image processing
import os
import io
import base64
import asyncio
import logging
from typing import Optional, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from diffusers import WanPipeline
import imageio
import b2sdk.v2 as b2
from contextlib import asynccontextmanager
import secrets
import hmac
import hashlib
import jwt
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
pipeline = None
device = None
b2_api = None
b2_bucket = None
thread_pool = ThreadPoolExecutor(max_workers=4)
redis_client = None

# Authentication configuration
JWT_SECRET = os.getenv('JWT_SECRET') or secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
API_KEY_HEADER = "X-API-Key"

# Authentication models
class User(BaseModel):
    user_id: str
    email: Optional[str] = None
    tier: str = "free"  # free, premium, enterprise
    daily_limit: int = 10
    monthly_limit: int = 100

class AuthToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600

# Enhanced Request/Response Models
class GenerationRequest(BaseModel):
    input_images: List[str] = Field(..., description="Base64 encoded images")
    prompt: str = Field(..., description="Generation prompt")
    negative_prompt: Optional[str] = Field("", description="Negative prompt")
    num_frames: int = Field(81, ge=25, le=125, description="Number of frames (25-125)")
    guidance_scale: float = Field(6.0, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    fps: int = Field(16, ge=8, le=30, description="Frames per second")
    resolution: str = Field("480p", description="Output resolution")
    adapter_strength: float = Field(1.0, ge=0.0, le=2.0, description="LoRA adapter strength")
    priority: Optional[str] = Field("normal", description="Job priority (normal, high)")

class GenerationResponse(BaseModel):
    status: str
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: float
    frames_generated: int
    processing_time: float
    cost_estimate: float
    credits_used: int
    remaining_credits: int
    error_message: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class ApiKeyRequest(BaseModel):
    name: str
    expires_days: Optional[int] = 30

# Authentication setup
security = HTTPBearer()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield
    await shutdown_event()

app = FastAPI(
    title="Kissing Video Generator API", 
    version="3.0.0",
    description="Secure API for generating kissing videos with authentication",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def startup_event():
    """Initialize the model and services"""
    global pipeline, device, b2_api, b2_bucket, redis_client
    
    logger.info("🚀 Starting secure inference server...")
    
    # Initialize Redis for rate limiting and user management
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD'),
        db=0,  # Different DB from job storage
        decode_responses=True
    )
    
    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("✅ Redis connected for authentication")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        # Continue without Redis (degraded mode)
        redis_client = None
    
    # GPU setup
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = "cpu"
        logger.warning("⚠️ No GPU detected, using CPU (will be very slow)")
    
    # Initialize Backblaze B2
    await initialize_b2()
    
    try:
        # Load the base model
        logger.info("📥 Loading Wan2.1 I2V model...")
        pipeline = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        
        # Load the kissing LoRA
        logger.info("💋 Loading kissing LoRA...")
        pipeline.load_lora_weights(
            "Remade-AI/kissing", 
            adapter_name="kissing",
            weight_name="kissing_30_epochs.safetensors"
        )
        
        # Move to GPU and optimize
        pipeline = pipeline.to(device)
        
        # Enable optimizations
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            pipeline.enable_xformers_memory_efficient_attention()
        
        if device == "cuda":
            pipeline.enable_model_cpu_offload()
        
        logger.info("✅ Model loaded successfully!")
        
        # Warm up
        await warmup_model()
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise e

async def shutdown_event():
    """Cleanup resources"""
    global pipeline, thread_pool, redis_client
    if pipeline:
        del pipeline
        torch.cuda.empty_cache()
    
    thread_pool.shutdown(wait=True)
    
    if redis_client:
        redis_client.close()
    
    logger.info("🛑 Server shutdown complete")

# Authentication functions
async def create_jwt_token(user_data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token for user"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode = {"user_data": user_data, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def verify_jwt_token(token: str) -> dict:
    """Verify JWT token and return user data"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_data = payload.get("user_data")
        if user_data is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_data
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def verify_api_key(api_key: str) -> dict:
    """Verify API key and return user data"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    
    try:
        # Get user data associated with API key
        user_data = redis_client.hgetall(f"api_key:{api_key}")
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Check if API key is expired
        if user_data.get('expires_at'):
            expires_at = datetime.fromisoformat(user_data['expires_at'])
            if datetime.utcnow() > expires_at:
                raise HTTPException(status_code=401, detail="API key expired")
        
        return {
            "user_id": user_data['user_id'],
            "email": user_data.get('email'),
            "tier": user_data.get('tier', 'free'),
            "daily_limit": int(user_data.get('daily_limit', 10)),
            "monthly_limit": int(user_data.get('monthly_limit', 100))
        }
    except Exception as e:
        logger.error(f"API key verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid API key")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    api_key: Optional[str] = Header(None, alias=API_KEY_HEADER)
) -> User:
    """Get current authenticated user from JWT token or API key"""
    
    # Try API key first
    if api_key:
        user_data = await verify_api_key(api_key)
        return User(**user_data)
    
    # Try JWT token
    if credentials:
        user_data = await verify_jwt_token(credentials.credentials)
        return User(**user_data)
    
    raise HTTPException(status_code=401, detail="Authentication required")

async def check_rate_limits(user: User) -> dict:
    """Check and update rate limits for user"""
    if not redis_client:
        # Degraded mode - allow requests but warn
        logger.warning("⚠️ Rate limiting unavailable - Redis not connected")
        return {"allowed": True, "daily_used": 0, "monthly_used": 0}
    
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        month = datetime.utcnow().strftime("%Y-%m")
        
        daily_key = f"rate_limit:daily:{user.user_id}:{today}"
        monthly_key = f"rate_limit:monthly:{user.user_id}:{month}"
        
        # Get current usage
        daily_used = int(redis_client.get(daily_key) or 0)
        monthly_used = int(redis_client.get(monthly_key) or 0)
        
        # Check limits
        if daily_used >= user.daily_limit:
            raise HTTPException(
                status_code=429, 
                detail=f"Daily limit exceeded ({user.daily_limit} requests per day)"
            )
        
        if monthly_used >= user.monthly_limit:
            raise HTTPException(
                status_code=429, 
                detail=f"Monthly limit exceeded ({user.monthly_limit} requests per month)"
            )
        
        # Increment counters
        pipe = redis_client.pipeline()
        pipe.incr(daily_key)
        pipe.expire(daily_key, 86400)  # 24 hours
        pipe.incr(monthly_key)
        pipe.expire(monthly_key, 2592000)  # 30 days
        pipe.execute()
        
        return {
            "allowed": True,
            "daily_used": daily_used + 1,
            "monthly_used": monthly_used + 1,
            "daily_remaining": user.daily_limit - daily_used - 1,
            "monthly_remaining": user.monthly_limit - monthly_used - 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        # Allow request if rate limiting fails
        return {"allowed": True, "daily_used": 0, "monthly_used": 0}

def calculate_credits_cost(request: GenerationRequest, user: User) -> int:
    """Calculate credit cost based on request parameters and user tier"""
    base_cost = 1
    
    # Frame-based pricing
    frame_multiplier = request.num_frames / 81
    
    # Resolution-based pricing
    resolution_multiplier = 1.5 if request.resolution == "720p" else 1.0
    
    # Priority-based pricing
    priority_multiplier = 2.0 if request.priority == "high" else 1.0
    
    # Tier-based discounts
    tier_discount = {
        "free": 1.0,
        "premium": 0.8,
        "enterprise": 0.6
    }.get(user.tier, 1.0)
    
    final_cost = int(base_cost * frame_multiplier * resolution_multiplier * priority_multiplier * tier_discount)
    return max(1, final_cost)  # Minimum 1 credit

# Refactored image processing with DRY principle
async def preprocess_single_image_async(
    image: Image.Image, 
    target_size: tuple = (512, 512),
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Advanced image preprocessing function - reusable for all image operations
    """
    def _preprocess():
        if maintain_aspect:
            # Calculate scaling to maintain aspect ratio
            img_ratio = image.width / image.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider than target
                new_width = target_size[0]
                new_height = int(target_size[0] / img_ratio)
            else:
                # Image is taller than target
                new_height = target_size[1]
                new_width = int(target_size[1] * img_ratio)
            
            # Resize image with high-quality resampling
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and center the resized image
            new_image = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            new_image.paste(resized, (paste_x, paste_y))
            
            return new_image
        else:
            # Direct resize without maintaining aspect ratio
            return image.resize(target_size, Image.Resampling.LANCZOS)
    
    return await asyncio.to_thread(_preprocess)

async def combine_couple_images_async(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """
    Refactored to use the superior preprocess_single_image_async function
    """
    def _combine():
        target_height = 512
        
        # Calculate aspect-preserving widths
        width1 = int(img1.width * target_height / img1.height)
        width2 = int(img2.width * target_height / img2.height)
        
        # Create combined canvas
        combined_width = width1 + width2
        combined_image = Image.new('RGB', (combined_width, target_height))
        
        # Resize images to target height while preserving aspect ratio
        img1_resized = img1.resize((width1, target_height), Image.Resampling.LANCZOS)
        img2_resized = img2.resize((width2, target_height), Image.Resampling.LANCZOS)
        
        # Paste images side by side
        combined_image.paste(img1_resized, (0, 0))
        combined_image.paste(img2_resized, (width1, 0))
        
        return combined_image
    
    # First combine, then use our superior preprocessing
    combined = await asyncio.to_thread(_combine)
    
    # Apply the same advanced preprocessing used for single images
    return await preprocess_single_image_async(combined, target_size=(512, 512))

# Keep original function name for backward compatibility
async def preprocess_image_async(image: Image.Image, target_size: tuple = (512, 512)) -> Image.Image:
    """Wrapper for backward compatibility"""
    return await preprocess_single_image_async(image, target_size)

# Authentication endpoints
@app.post("/auth/login", response_model=AuthToken)
async def login(request: LoginRequest):
    """Login endpoint - simplified for demo"""
    # In production, verify against your user database
    if request.email == "demo@example.com" and request.password == "demo123":
        user_data = {
            "user_id": "demo_user",
            "email": request.email,
            "tier": "premium",
            "daily_limit": 50,
            "monthly_limit": 500
        }
        
        access_token = await create_jwt_token(user_data)
        return AuthToken(access_token=access_token)
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/auth/api-key")
async def create_api_key(
    request: ApiKeyRequest,
    current_user: User = Depends(get_current_user)
):
    """Create API key for user"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="API key service unavailable")
    
    # Generate secure API key
    api_key = f"gvk_{secrets.token_urlsafe(32)}"
    
    # Calculate expiration
    expires_at = datetime.utcnow() + timedelta(days=request.expires_days or 30)
    
    # Store in Redis
    redis_client.hset(f"api_key:{api_key}", mapping={
        "user_id": current_user.user_id,
        "email": current_user.email or "",
        "tier": current_user.tier,
        "daily_limit": current_user.daily_limit,
        "monthly_limit": current_user.monthly_limit,
        "name": request.name,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": expires_at.isoformat()
    })
    
    # Set expiration on the Redis key
    redis_client.expire(f"api_key:{api_key}", int((expires_at - datetime.utcnow()).total_seconds()))
    
    return {
        "api_key": api_key,
        "name": request.name,
        "expires_at": expires_at.isoformat(),
        "usage_instructions": {
            "header": API_KEY_HEADER,
            "example": f"curl -H '{API_KEY_HEADER}: {api_key}' ..."
        }
    }

@app.get("/auth/me")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile and usage stats"""
    rate_info = await check_rate_limits(current_user)
    
    return {
        "user": current_user.dict(),
        "usage": {
            "daily_used": rate_info.get("daily_used", 0),
            "daily_remaining": rate_info.get("daily_remaining", current_user.daily_limit),
            "monthly_used": rate_info.get("monthly_used", 0),
            "monthly_remaining": rate_info.get("monthly_remaining", current_user.monthly_limit)
        }
    }

# Initialize B2 (same as before)
async def initialize_b2():
    """Initialize Backblaze B2 client"""
    global b2_api, b2_bucket
    
    try:
        application_key_id = os.getenv('B2_APPLICATION_KEY_ID')
        application_key = os.getenv('B2_APPLICATION_KEY')
        bucket_name = os.getenv('B2_BUCKET_NAME', 'kissing-videos')
        
        if not all([application_key_id, application_key]):
            raise ValueError("B2 credentials not found in environment variables")
        
        info = b2.InMemoryAccountInfo()
        b2_api = b2.B2Api(info)
        b2_api.authorize_account("production", application_key_id, application_key)
        
        b2_bucket = b2_api.get_bucket_by_name(bucket_name)
        
        logger.info(f"✅ Backblaze B2 initialized with bucket: {bucket_name}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize B2: {str(e)}")
        raise e

# Video processing functions (same as before but with imageio)
async def process_frames_to_video_bytes_imageio(frames: List[np.ndarray], fps: int) -> bytes:
    """Convert frames to MP4 video bytes using imageio (no disk I/O)"""
    def _process_video():
        try:
            buffer = io.BytesIO()
            
            processed_frames = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    processed_frames.append(frame)
                else:
                    if len(frame.shape) == 2:
                        frame = np.stack([frame] * 3, axis=-1)
                    processed_frames.append(frame)
            
            with imageio.get_writer(
                buffer, 
                format='mp4', 
                mode='I',
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p',
                macro_block_size=1
            ) as writer:
                for frame in processed_frames:
                    writer.append_data(frame)
            
            video_bytes = buffer.getvalue()
            buffer.close()
            
            return video_bytes
            
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            raise e
    
    return await asyncio.to_thread(_process_video)

async def create_thumbnail_bytes_imageio(frames: List[np.ndarray], timestamp_frame: int = 10) -> bytes:
    """Create thumbnail using imageio for consistency"""
    def _create_thumbnail():
        try:
            frame_idx = min(timestamp_frame, len(frames) - 1)
            frame = frames[frame_idx]
            
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            height, width = frame.shape[:2]
            target_width = 480
            target_height = int(height * target_width / width)
            
            image = Image.fromarray(frame)
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            resized_frame = np.array(image)
            
            buffer = io.BytesIO()
            imageio.imwrite(buffer, resized_frame, format='JPEG', quality=85)
            
            thumbnail_bytes = buffer.getvalue()
            buffer.close()
            
            return thumbnail_bytes
            
        except Exception as e:
            logger.error(f"Thumbnail creation error: {str(e)}")
            raise e
    
    return await asyncio.to_thread(_create_thumbnail)

async def upload_to_b2(data: bytes, filename: str, content_type: str) -> str:
    """Upload bytes data to Backblaze B2 with retry logic"""
    def _upload():
        max_retries = 3
        for attempt in range(max_retries):
            try:
                file_info = b2_bucket.upload_bytes(
                    data,
                    filename,
                    content_type=content_type,
                    file_infos={
                        'timestamp': str(int(datetime.now().timestamp())),
                        'size': str(len(data))
                    }
                )
                
                download_url = b2_api.get_download_url_for_fileid(file_info.id_)
                return download_url
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Upload attempt {attempt + 1} failed: {str(e)}")
                import time
                time.sleep(2 ** attempt)
    
    return await asyncio.to_thread(_upload)

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image with validation"""
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        
        if len(image_data) < 100:
            raise ValueError("Image data too small")
        
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if image.width < 64 or image.height < 64:
            raise ValueError("Image dimensions too small (minimum 64x64)")
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

async def warmup_model():
    """Warm up model with proper LoRA adapter usage"""
    try:
        test_image = Image.new('RGB', (512, 512), color='blue')
        test_prompt = "A couple standing together, they are k144ing kissing"
        
        pipeline.set_adapters(["kissing"], adapter_weights=[1.0])
        
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = pipeline(
                image=test_image,
                prompt=test_prompt,
                num_frames=25,
                guidance_scale=6.0,
                height=480,
                width=480,
                generator=generator
            )
        
        logger.info("✅ Model warmup completed")
        del output
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"⚠️ Warmup failed: {str(e)}")

# Secured main generation endpoint
@app.post("/generate", response_model=GenerationResponse)
async def generate_video(
    request: GenerationRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate kissing video with authentication and rate limiting"""
    start_time = datetime.now()

    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check rate limits
    rate_info = await check_rate_limits(current_user)

    # Calculate credit cost
    credits_cost = calculate_credits_cost(request, current_user)

    try:
        logger.info(f"🎬 Starting authenticated video generation for user {current_user.user_id}")
        logger.info(f"💳 Credits cost: {credits_cost}")

        # Decode and preprocess images using refactored functions
        images = []
        for i, img_b64 in enumerate(request.input_images):
            try:
                image = decode_base64_image(img_b64)
                image = await preprocess_single_image_async(image)
                images.append(image)
                logger.info(f"✅ Processed image {i+1}/{len(request.input_images)}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing image {i+1}: {str(e)}")

        # Combine images using refactored function
        if len(images) == 2:
            input_image = await combine_couple_images_async(images[0], images[1])
        elif len(images) == 1:
            input_image = images[0]
        else:
            raise HTTPException(status_code=400, detail="Please provide 1 or 2 images")

        # Enhanced prompt
        enhanced_prompt = f"{request.prompt.strip()}, they are k144ing kissing"
        if not any(word in request.prompt.lower() for word in ['kiss', 'k144ing']):
            enhanced_prompt = f"A couple {request.prompt.strip()}, they are k144ing kissing"

        logger.info(f"📝 Using prompt: {enhanced_prompt}")

        # Set LoRA adapter with proper strength control
        pipeline.load_lora_weights("Remade-AI/kissing", adapter_name="kissing", weight_name="kissing_30_epochs..safetensors")
        pipeline.set_adapters(["kissing"], adapter_weights=[request.adapter_strength])
        logger.info(f"🎛️ Set LoRA adapter strength: {request.adapter_strength}")

        # Create isolated generator for seed
        generator = None
        if request.seed:
            generator = torch.Generator(device=device).manual_seed(request.seed)
            logger.info(f"🎲 Using seed: {request.seed}")

        # Generate video
        logger.info("🎥 Generating video...")
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = pipeline(
                image=input_image,
                prompt=enhanced_prompt,
                negative_prompt=request.negative_prompt,
                num_frames=request.num_frames,
                guidance_scale=request.guidance_scale,
                height=480 if request.resolution == "480p" else 720,
                width=480 if request.resolution == "480p" else 720,
                num_inference_steps=30,
                generator=generator
            )

        # Process frames with imageio
        frames = output.frames[0]
        frames_np = [np.array(frame) for frame in frames]

        logger.info("🎞️ Converting frames to video with imageio...")
        video_bytes = await process_frames_to_video_bytes_imageio(frames_np, request.fps)

        logger.info("📸 Creating thumbnail with imageio...")
        thumbnail_bytes = await create_thumbnail_bytes_imageio(frames_np)

        # Upload to B2
        timestamp = int(datetime.now().timestamp())
        video_filename = f"videos/{current_user.user_id}/kissing_video_{timestamp}.mp4"
        thumbnail_filename = f"thumbnails/{current_user.user_id}/thumb_{timestamp}.jpg"

        logger.info("☁️ Uploading to Backblaze B2...")
        video_url, thumbnail_url = await asyncio.gather(
            upload_to_b2(video_bytes, video_filename, "video/mp4"),
            upload_to_b2(thumbnail_bytes, thumbnail_filename, "image/jpeg")
        )

        total_time = (datetime.now() - start_time).total_seconds()

        monthly_remaining = rate_info.get("monthly_remaining", 0)

        return GenerationResponse(
            status="success",
            video_url=video_url,
            thumbnail_url=thumbnail_url,
            duration=float(len(frames_np) / request.fps),
            frames_generated=len(frames_np),
            processing_time=total_time,
            cost_estimate=float(credits_cost),
            credits_used=credits_cost,
            remaining_credits=monthly_remaining,
        )

    except Exception as e:
        logger.error(f"❌ Video generation failed for user {current_user.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during video generation: {str(e)}")

    finally:
        # This block always runs, ensuring the GPU cache is cleared.
        logger.info("🧹 Clearing GPU cache.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
