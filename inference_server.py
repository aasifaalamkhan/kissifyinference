# inference_server.py - With authentication, refactored image processing, and manual IP-Adapter loading
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
# --- MODIFIED IMPORTS ---
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, IPAdapterModel
from transformers import CLIPVisionModel, CLIPImageProcessor
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
    tier: str = "free"
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
    ip_adapter_scale: float = Field(0.8, ge=0.0, le=1.5, description="IP-Adapter strength")
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
    version="3.2.0",
    description="Secure API for generating kissing videos with manual IP-Adapter loading, authentication and debug features",
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
    global pipeline, device, b2_api, b2_bucket, redis_client
    logger.info("🚀 Starting secure inference server...")
    
    redis_client = redis.Redis( host=os.getenv('REDIS_HOST', 'localhost'), port=int(os.getenv('REDIS_PORT', 6379)), password=os.getenv('REDIS_PASSWORD'), db=0, decode_responses=True )
    try:
        redis_client.ping()
        logger.info("✅ Redis connected for authentication")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        redis_client = None

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = "cpu"
        logger.warning("⚠️ No GPU detected, using CPU (will be very slow)")
    
    await initialize_b2()
    
    try:
        model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        
        logger.info("📥 Loading Image Encoder (CLIP)...")
        image_encoder = CLIPVisionModel.from_pretrained( model_id, subfolder="image_encoder", torch_dtype=torch.bfloat16 )
        
        logger.info("📥 Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained( model_id, subfolder="vae", torch_dtype=torch.float32 )
        
        logger.info("📥 Loading Main I2V Pipeline...")
        pipeline = WanImageToVideoPipeline.from_pretrained( model_id, image_encoder=image_encoder, vae=vae, torch_dtype=torch.bfloat16 )
        
        logger.info("💋 Loading kissing LoRA...")
        pipeline.load_lora_weights( "Remade-AI/kissing", adapter_name="kissing", weight_name="kissing_30_epochs.safetensors" )
        
        # --- MODIFIED IP-ADAPTER LOADING ---
        logger.info("🎨 Manually loading IP-Adapter components...")
        # Since the pipeline doesn't have a built-in .load_ip_adapter(), we load the components manually.
        ip_adapter_id = "h94/IP-Adapter"
        pipeline.ip_adapter = IPAdapterModel.from_pretrained(ip_adapter_id, subfolder="models", weight_name="ip-adapter-plus_sd15.bin", torch_dtype=torch.bfloat16)
        # We need the specific image processor that corresponds to the IP-Adapter's vision encoder
        pipeline.image_processor_ip = CLIPImageProcessor.from_pretrained(model_id, subfolder="image_encoder")
        # --- END MODIFICATION ---

        pipeline.to(device)
        pipeline.ip_adapter.to(device)

        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            pipeline.enable_xformers_memory_efficient_attention()
        if device == "cuda":
            pipeline.enable_model_cpu_offload()

        logger.info("✅ Model loaded successfully!")
        await warmup_model()
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise e

async def shutdown_event():
    global pipeline, thread_pool, redis_client
    if pipeline:
        del pipeline
        torch.cuda.empty_cache()
    thread_pool.shutdown(wait=True)
    if redis_client:
        redis_client.close()
    logger.info("🛑 Server shutdown complete")

# ... (Authentication, B2, and other utility functions remain unchanged) ...
async def create_jwt_token(user_data: dict, expires_delta: Optional[timedelta] = None):
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    to_encode = {"user_data": user_data, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def verify_jwt_token(token: str) -> dict:
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
    if not redis_client:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    try:
        user_data = redis_client.hgetall(f"api_key:{api_key}")
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid API key")
        if user_data.get('expires_at'):
            expires_at = datetime.fromisoformat(user_data['expires_at'])
            if datetime.utcnow() > expires_at:
                raise HTTPException(status_code=401, detail="API key expired")
        return { "user_id": user_data['user_id'], "email": user_data.get('email'), "tier": user_data.get('tier', 'free'), "daily_limit": int(user_data.get('daily_limit', 10)), "monthly_limit": int(user_data.get('monthly_limit', 100)) }
    except Exception as e:
        logger.error(f"API key verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid API key")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), api_key: Optional[str] = Header(None, alias=API_KEY_HEADER)) -> User:
    if api_key:
        user_data = await verify_api_key(api_key)
        return User(**user_data)
    if credentials:
        user_data = await verify_jwt_token(credentials.credentials)
        return User(**user_data)
    raise HTTPException(status_code=401, detail="Authentication required")

async def check_rate_limits(user: User) -> dict:
    if not redis_client:
        logger.warning("⚠️ Rate limiting unavailable - Redis not connected")
        return {"allowed": True, "daily_used": 0, "monthly_used": 0}
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        month = datetime.utcnow().strftime("%Y-%m")
        daily_key = f"rate_limit:daily:{user.user_id}:{today}"
        monthly_key = f"rate_limit:monthly:{user.user_id}:{month}"
        daily_used = int(redis_client.get(daily_key) or 0)
        monthly_used = int(redis_client.get(monthly_key) or 0)
        if daily_used >= user.daily_limit:
            raise HTTPException(status_code=429, detail=f"Daily limit exceeded ({user.daily_limit} requests per day)")
        if monthly_used >= user.monthly_limit:
            raise HTTPException(status_code=429, detail=f"Monthly limit exceeded ({user.monthly_limit} requests per month)")
        pipe = redis_client.pipeline()
        pipe.incr(daily_key)
        pipe.expire(daily_key, 86400)
        pipe.incr(monthly_key)
        pipe.expire(monthly_key, 2592000)
        pipe.execute()
        return { "allowed": True, "daily_used": daily_used + 1, "monthly_used": monthly_used + 1, "daily_remaining": user.daily_limit - daily_used - 1, "monthly_remaining": user.monthly_limit - monthly_used - 1 }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        return {"allowed": True, "daily_used": 0, "monthly_used": 0}

def calculate_credits_cost(request: GenerationRequest, user: User) -> int:
    base_cost = 1
    frame_multiplier = request.num_frames / 81
    resolution_multiplier = 1.5 if request.resolution == "720p" else 1.0
    priority_multiplier = 2.0 if request.priority == "high" else 1.0
    tier_discount = { "free": 1.0, "premium": 0.8, "enterprise": 0.6 }.get(user.tier, 1.0)
    final_cost = int(base_cost * frame_multiplier * resolution_multiplier * priority_multiplier * tier_discount)
    return max(1, final_cost)

async def preprocess_single_image_async(image: Image.Image, target_size: tuple = (512, 512), maintain_aspect: bool = True) -> Image.Image:
    def _preprocess():
        if maintain_aspect:
            img_ratio = image.width / image.height
            target_ratio = target_size[0] / target_size[1]
            if img_ratio > target_ratio:
                new_width = target_size[0]
                new_height = int(target_size[0] / img_ratio)
            else:
                new_height = target_size[1]
                new_width = int(target_size[1] * img_ratio)
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            new_image = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            new_image.paste(resized, (paste_x, paste_y))
            return new_image
        else:
            return image.resize(target_size, Image.Resampling.LANCZOS)
    return await asyncio.to_thread(_preprocess)

async def stitch_images_side_by_side_async(img1: Image.Image, img2: Image.Image) -> Image.Image:
    def _stitch():
        target_height = 512
        width1 = int(img1.width * target_height / img1.height)
        width2 = int(img2.width * target_height / img2.height)
        combined_width = width1 + width2
        combined_image = Image.new('RGB', (combined_width, target_height))
        img1_resized = img1.resize((width1, target_height), Image.Resampling.LANCZOS)
        img2_resized = img2.resize((width2, target_height), Image.Resampling.LANCZOS)
        combined_image.paste(img1_resized, (0, 0))
        combined_image.paste(img2_resized, (width1, 0))
        return combined_image
    return await asyncio.to_thread(_stitch)

@app.post("/auth/login", response_model=AuthToken)
async def login(request: LoginRequest):
    if request.email == "demo@example.com" and request.password == "demo123":
        user_data = { "user_id": "demo_user", "email": request.email, "tier": "premium", "daily_limit": 50, "monthly_limit": 500 }
        access_token = await create_jwt_token(user_data)
        return AuthToken(access_token=access_token)
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/auth/api-key")
async def create_api_key(request: ApiKeyRequest, current_user: User = Depends(get_current_user)):
    if not redis_client:
        raise HTTPException(status_code=503, detail="API key service unavailable")
    api_key = f"gvk_{secrets.token_urlsafe(32)}"
    expires_at = datetime.utcnow() + timedelta(days=request.expires_days or 30)
    redis_client.hset(f"api_key:{api_key}", mapping={ "user_id": current_user.user_id, "email": current_user.email or "", "tier": current_user.tier, "daily_limit": current_user.daily_limit, "monthly_limit": current_user.monthly_limit, "name": request.name, "created_at": datetime.utcnow().isoformat(), "expires_at": expires_at.isoformat() })
    redis_client.expire(f"api_key:{api_key}", int((expires_at - datetime.utcnow()).total_seconds()))
    return { "api_key": api_key, "name": request.name, "expires_at": expires_at.isoformat(), "usage_instructions": { "header": API_KEY_HEADER, "example": f"curl -H '{API_KEY_HEADER}: {api_key}' ..." } }

@app.get("/auth/me")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    rate_info = await check_rate_limits(current_user)
    return { "user": current_user.dict(), "usage": { "daily_used": rate_info.get("daily_used", 0), "daily_remaining": rate_info.get("daily_remaining", current_user.daily_limit), "monthly_used": rate_info.get("monthly_used", 0), "monthly_remaining": rate_info.get("monthly_remaining", current_user.monthly_limit) } }

async def initialize_b2():
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

async def process_frames_to_video_bytes_imageio(frames: List[np.ndarray], fps: int) -> bytes:
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
            with imageio.get_writer(buffer, format='mp4', mode='I', fps=fps, codec='libx264', quality=8, pixelformat='yuv420p') as writer:
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
    def _upload():
        max_retries = 3
        for attempt in range(max_retries):
            try:
                file_info = b2_bucket.upload_bytes(data, filename, content_type=content_type, file_infos={ 'timestamp': str(int(datetime.now().timestamp())), 'size': str(len(data)) })
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

async def convert_pil_to_bytes_async(image: Image.Image, format: str = "JPEG") -> bytes:
    def _convert():
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95)
        return buffer.getvalue()
    return await asyncio.to_thread(_convert)


async def warmup_model():
    try:
        logger.info("🔥 Warming up model with manual IP-Adapter logic...")
        test_image = Image.new('RGB', (512, 512), color='blue')
        test_prompt = "A couple standing together, they are k144ing kissing"

        pipeline.set_adapters(["kissing"], adapter_weights=[1.0])
        
        # --- MODIFIED WARMUP FOR MANUAL IP-ADAPTER ---
        ip_image = pipeline.image_processor_ip(test_image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)
        image_embeds = pipeline.ip_adapter.get_image_embeds(ip_image)
        scaled_embeds = image_embeds * 0.8 # Use default scale for warmup
        
        negative_embeds = torch.zeros_like(scaled_embeds)
        final_embeds = torch.cat([scaled_embeds, negative_embeds])
        
        cross_attention_kwargs = {"ip_adapter_image_embeds": final_embeds}
        # --- END MODIFICATION ---

        generator = torch.Generator(device=device).manual_seed(42)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = pipeline(
                image=test_image,
                prompt=test_prompt,
                cross_attention_kwargs=cross_attention_kwargs,
                num_frames=40,
                guidance_scale=7.5,
                height=480,
                width=480,
                generator=generator
            )

        logger.info("✅ Model warmup completed")
        del output
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"⚠️ Warmup failed: {str(e)}")

@app.post("/generate", response_model=GenerationResponse)
async def generate_video(request: GenerationRequest, current_user: User = Depends(get_current_user)):
    global pipeline
    start_time = datetime.now()

    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rate_info = await check_rate_limits(current_user)
    credits_cost = calculate_credits_cost(request, current_user)

    try:
        logger.info(f"🎬 Starting authenticated video generation for user {current_user.user_id}")
        logger.info(f"💳 Credits cost: {credits_cost}")

        timestamp = int(datetime.now().timestamp())
        images = [decode_base64_image(img_b64) for i, img_b64 in enumerate(request.input_images)]
        
        if len(images) == 2:
            stitched_image = await stitch_images_side_by_side_async(images[0], images[1])
            raw_bytes = await convert_pil_to_bytes_async(stitched_image)
            asyncio.create_task(upload_to_b2(raw_bytes, f"debug/{current_user.user_id}/stitched_raw_{timestamp}.jpg", "image/jpeg"))
            input_image = await preprocess_single_image_async(stitched_image)
        elif len(images) == 1:
            input_image = await preprocess_single_image_async(images[0])
        else:
            raise HTTPException(status_code=400, detail="Please provide 1 or 2 input images.")

        processed_bytes = await convert_pil_to_bytes_async(input_image)
        asyncio.create_task(upload_to_b2(processed_bytes, f"debug/{current_user.user_id}/final_input_{timestamp}.jpg", "image/jpeg"))

        user_context = request.prompt.strip()
        for word in ["kissing", "k144ing", "kiss", "a couple", "a man and a woman"]:
            user_context = user_context.lower().replace(word, "")
        enhanced_prompt = f"masterpiece, best quality, high resolution, A man and a woman, {user_context.strip()}, are embracing each other closely. They then engage in a passionate and intimate k144ing kissing scene."
        logger.info(f"📝 Using new enhanced prompt: {enhanced_prompt}")

        pipeline.set_adapters(["kissing"], adapter_weights=[request.adapter_strength])
        logger.info(f"🎛️ Set LoRA adapter strength: {request.adapter_strength}")

        # --- MODIFIED FOR MANUAL IP-ADAPTER ---
        logger.info(f"🎨 Processing image for IP-Adapter with scale: {request.ip_adapter_scale}")
        # 1. Process the image with the IP-Adapter's specific processor
        ip_image = pipeline.image_processor_ip(input_image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)
        
        # 2. Get the image embeddings from the IP-Adapter model
        image_embeds = pipeline.ip_adapter.get_image_embeds(ip_image)

        # 3. Scale the embeddings and create negative embeddings for classifier-free guidance
        scaled_embeds = image_embeds * request.ip_adapter_scale
        negative_embeds = torch.zeros_like(scaled_embeds)

        # The pipeline expects the embeddings to be concatenated (positive, negative)
        final_embeds = torch.cat([scaled_embeds, negative_embeds])

        # 4. Pass the embeddings through cross_attention_kwargs
        cross_attention_kwargs = {"ip_adapter_image_embeds": final_embeds}
        # --- END MODIFICATION ---

        generator = None
        if request.seed:
            generator = torch.Generator(device=device).manual_seed(request.seed)
            logger.info(f"🎲 Using seed: {request.seed}")

        logger.info("🎥 Generating video...")
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = pipeline(
                image=input_image,
                prompt=enhanced_prompt,
                negative_prompt=request.negative_prompt,
                cross_attention_kwargs=cross_attention_kwargs, # Pass the custom embeddings here
                num_frames=request.num_frames,
                guidance_scale=request.guidance_scale,
                height=480 if request.resolution == "480p" else 720,
                width=480 if request.resolution == "480p" else 720,
                num_inference_steps=30,
                generator=generator
            )

        frames = output.frames[0]
        frames_np = [np.array(frame) for frame in frames]
        video_bytes = await process_frames_to_video_bytes_imageio(frames_np, request.fps)
        thumbnail_bytes = await create_thumbnail_bytes_imageio(frames_np)

        video_filename = f"videos/{current_user.user_id}/kissing_video_{timestamp}.mp4"
        thumbnail_filename = f"thumbnails/{current_user.user_id}/thumb_{timestamp}.jpg"

        video_url, thumbnail_url = await asyncio.gather(
            upload_to_b2(video_bytes, video_filename, "video/mp4"),
            upload_to_b2(thumbnail_bytes, thumbnail_filename, "image/jpeg")
        )

        total_time = (datetime.now() - start_time).total_seconds()
        monthly_remaining = rate_info.get("monthly_remaining", 0)

        return GenerationResponse(
            status="success", video_url=video_url, thumbnail_url=thumbnail_url, duration=float(len(frames_np) / request.fps),
            frames_generated=len(frames_np), processing_time=total_time, cost_estimate=float(credits_cost),
            credits_used=credits_cost, remaining_credits=monthly_remaining,
        )

    except Exception as e:
        logger.error(f"❌ Video generation failed for user {current_user.user_id}: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

    finally:
        logger.info("🧹 Clearing GPU cache.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
