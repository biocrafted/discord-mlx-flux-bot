import discord
from discord.ext import commands, tasks
import asyncio
import os
import shutil
import uuid
import time
import subprocess
import re # For parsing flags
from PIL import Image, ImageOps

# Assuming flux module is available (as in txt2image.py)
# You'll need to ensure the flux library and its models are correctly installed and accessible.
try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from tqdm import tqdm # Used in original script, can be optional for bot
    from flux import FluxPipeline # This is a key dependency from your txt2image.py context
except ImportError as e:
    print(f"Critical Import Error: {e}. Please ensure MLX and Flux and their dependencies are installed.")
    # Depending on your setup, you might want to sys.exit() here if these are crucial at startup.

# Background removal imports (from oldbot.py)
try:
    from rembg import remove, new_session
    from transparent_background import Remover as HighQualityRemover
except ImportError:
    print("Warning: Background removal libraries (rembg, transparent-background) not found. Background removal will be disabled.")
    # Define dummy functions or disable the feature if imports fail
    new_session = None
    HighQualityRemover = None
    remove = None


# --- BOT CONFIGURATION ---
BOT_TOKEN = os.getenv("IMAGEBOT_DISCORD_TOKEN", "YOUR_BOT_TOKEN")
COMMAND_PREFIX = "$img" # Or any prefix you prefer for commands, or rely on mentions

# --- DIRECTORY CONFIGURATION ---
PROCESSING_BASE_DIR = "imagebot_processing_jobs"
MAX_CONCURRENT_JOBS = 1 # Limit to 1 due to potential high resource usage of image generation

# --- DEFAULT GENERATION PARAMETERS ---
DEFAULT_MODEL_NAME = "schnell" # "schnell" or "dev"
DEFAULT_N_IMAGES = 1 # Bot will generate 1 image by default
DEFAULT_IMAGE_SIZE = (512, 512)
DEFAULT_STEPS = 2 # As per your request for "schnell"
DEFAULT_GUIDANCE = 4.0
DEFAULT_T5_PADDING = True # From txt2image.py args

# --- DEFAULT PROCESSING PARAMETERS ---
DEFAULT_PIXELATE_FACTOR = 12
DEFAULT_QUANTIZE_COLORS = 16
DEFAULT_SKIP_BG_REMOVAL = False
DEFAULT_HQ_BG_REMOVAL = False

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True # Allow DMs

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents, case_insensitive=True)
processing_queue = asyncio.Queue()

# --- HELPER: FFMPEG CHECK (Not directly used by txt2image, but good practice from oldbot) ---
def check_ffmpeg(): # Retained for potential future use or consistency
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found. Not essential for core image generation, but might be for other utilities.")
        return False
    print("ffmpeg found.")
    return True

# --- HELPER: SUBPROCESS (Retained for potential future use) ---
async def run_subprocess_async(cmd_list, job_dir_log_path, log_filename="subprocess_log.txt"):
    process = await asyncio.create_subprocess_exec(
        *cmd_list,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    log_file_path = os.path.join(job_dir_log_path, log_filename)
    with open(log_file_path, "a") as log_file:
        if stdout:
            log_file.write("--- STDOUT ---\n")
            log_file.write(stdout.decode(errors='ignore'))
        if stderr:
            log_file.write("--- STDERR ---\n")
            log_file.write(stderr.decode(errors='ignore'))

    if process.returncode != 0:
        error_message = f"Command '{' '.join(cmd_list)}' failed with exit code {process.returncode}. See {log_file_path}"
        print(error_message)
        raise subprocess.CalledProcessError(process.returncode, cmd_list, output=stdout, stderr=stderr)
    return True

# --- Function to create directories for a job ---
def create_job_directories(job_id):
    job_dir = os.path.join(PROCESSING_BASE_DIR, job_id)
    # Define directories for each stage
    generated_raw_dir = os.path.join(job_dir, "00_generated_raw")
    no_bg_dir = os.path.join(job_dir, "01_no_bg")
    pixelated_dir = os.path.join(job_dir, "02_pixelated")
    final_quantized_dir = os.path.join(job_dir, "03_final_quantized")

    for dir_path in [job_dir, generated_raw_dir, no_bg_dir, pixelated_dir, final_quantized_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    return job_dir, generated_raw_dir, no_bg_dir, pixelated_dir, final_quantized_dir

# --- CORE IMAGE GENERATION LOGIC (Adapted from txt2image.py) ---
def to_latent_size(image_size): # Helper from txt2image.py
    h, w = image_size
    h_new = ((h + 15) // 16) * 16
    w_new = ((w + 15) // 16) * 16
    if (h_new, w_new) != image_size:
        print(f"Warning: Image dimensions adjusted to be divisible by 16px: {h_new}x{w_new}.")
    return (h_new // 8, w_new // 8), (h_new, w_new)

async def generate_image_with_flux(
    prompt_text: str,
    output_image_path: str,
    job_id: str, # For logging
    model_name: str = DEFAULT_MODEL_NAME,
    n_images: int = DEFAULT_N_IMAGES,
    image_size: tuple = DEFAULT_IMAGE_SIZE,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    seed: int = None,
    t5_padding: bool = DEFAULT_T5_PADDING
):
    print(f"Job {job_id}: Starting image generation with FLUX...")
    try:
        flux_pipeline = FluxPipeline(f"flux-{model_name}", t5_padding=t5_padding)
        
        actual_steps = steps if model_name == "dev" else (steps or 2) # txt2image logic for steps

        # Distributed generation setup (simplified for single bot instance)
        # group = mx.distributed.init() # For a bot, usually not distributed like this
        # For now, assume single process, single device.
        # If sharding or complex distributed setup is needed, this part needs more thought for a bot.

        if seed is None:
            seed = int(time.time() * 1000) % (2**32 -1) # Simple seed
        mx.random.seed(seed)
        np.random.seed(seed)
        print(f"Job {job_id}: Using seed {seed}")

        flux_pipeline.ensure_models_are_loaded() # Important for first run or if models are lazy-loaded

        latent_img_size, final_image_size = to_latent_size(image_size)

        print(f"Job {job_id}: Generating latents for prompt: '{prompt_text}' with size {final_image_size}, steps {actual_steps}...")
        latents_generator = flux_pipeline.generate_latents(
            prompt_text,
            n_images=n_images, # Should be 1 for the bot per call
            num_steps=actual_steps,
            latent_size=latent_img_size,
            guidance=guidance,
            seed=seed, # Pass the specific seed
        )

        # Conditioning
        conditioning = next(latents_generator)
        mx.eval(conditioning)
        # peak_mem_conditioning = mx.get_peak_memory() / 1024**3
        mx.reset_peak_memory()
        print(f"Job {job_id}: Conditioning complete.")

        # Memory optimization from txt2image.py
        del flux_pipeline.t5
        del flux_pipeline.clip

        # Denoising loop
        x_t_final = None
        # Wrap latents_generator with tqdm if detailed progress is desired in logs,
        # but for bot, simple print statements might be enough.
        print(f"Job {job_id}: Starting denoising loop for {actual_steps} steps...")
        for i, x_t in enumerate(latents_generator): # Use enumerate for step count if tqdm is removed
            mx.eval(x_t)
            x_t_final = x_t
            # print(f"Job {job_id}: Denoising step {i+1}/{actual_steps} complete.")
        print(f"Job {job_id}: Denoising loop finished.")


        # Memory optimization
        del flux_pipeline.flow
        # peak_mem_generation = mx.get_peak_memory() / 1024**3
        mx.reset_peak_memory()

        # Decode into images
        # Assuming n_images = 1, so decoding_batch_size = 1 effectively
        print(f"Job {job_id}: Decoding latents...")
        decoded_images = flux_pipeline.decode(x_t_final, latent_img_size)
        mx.eval(decoded_images)
        # peak_mem_decoding = mx.get_peak_memory() / 1024**3

        # Process and save the single image
        # The output 'decoded_images' is expected to be a tensor for a batch of images.
        # If n_images is 1, we take the first one.
        if decoded_images is None or len(decoded_images) == 0:
            raise ValueError("Image decoding resulted in no images.")

        single_image_tensor = decoded_images[0] # Get the first (and only) image
        
        # Rescale to 0-255 and convert to uint8
        single_image_np = (np.array(single_image_tensor) * 255).astype(np.uint8)
        
        # Create PIL Image
        pil_image = Image.fromarray(single_image_np)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        pil_image.save(output_image_path)
        print(f"Job {job_id}: Image generated and saved to {output_image_path}")
        return output_image_path, final_image_size # Return path and actual size used

    except Exception as e:
        print(f"Job {job_id}: ERROR during image generation: {e}")
        import traceback
        traceback.print_exc()
        raise # Re-raise to be caught by the pipeline executor

# --- IMAGE PROCESSING FUNCTIONS (Adapted from oldbot.py for single images) ---

async def process_image_remove_background_async(input_image_path, output_image_path, job_id, use_high_quality_model=False):
    print(f"Job {job_id}: Starting background removal. High quality: {use_high_quality_model}")
    if not (HighQualityRemover and new_session and remove):
        print(f"Job {job_id}: Background removal libraries not available. Skipping.")
        shutil.copy(input_image_path, output_image_path) # Just copy if libs missing
        return True

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    try:
        if use_high_quality_model:
            print(f"Job {job_id}: Initializing InSPyReNet model (High Quality)...")
            remover_instance = HighQualityRemover() # Consider initializing once if bot handles many HQ requests
            print(f"Job {job_id}: HQ Model initialized. Processing {input_image_path}...")
            
            def _process_hq_sync():
                img = Image.open(input_image_path).convert('RGB')
                out_img = remover_instance.process(img)
                out_img.save(output_image_path, "PNG")
            await asyncio.to_thread(_process_hq_sync)
        else:
            print(f"Job {job_id}: Initializing rembg session (u2net - Default Quality)...")
            session = new_session(model_name="u2net") # Consider session management for performance
            print(f"Job {job_id}: rembg session initialized. Processing {input_image_path}...")

            def _process_default_q_sync():
                with open(input_image_path, 'rb') as i_file:
                    input_data = i_file.read()
                output_data = remove(input_data, session=session)
                with open(output_image_path, 'wb') as o_file:
                    o_file.write(output_data)
            await asyncio.to_thread(_process_default_q_sync)
        
        print(f"Job {job_id}: Background removal complete. Output: {output_image_path}")
        return True
    except Exception as e:
        print(f"Job {job_id}: Error removing background from {input_image_path}: {e}")
        # Fallback: copy original image if removal fails
        try:
            shutil.copy(input_image_path, output_image_path)
        except Exception as e_copy:
            print(f"Job {job_id}: Failed to even copy original image after BG removal error: {e_copy}")
            return False
        return False # Indicate failure

async def process_image_pixelate_async(input_image_path, output_image_path, job_id, downscale_factor):
    print(f"Job {job_id}: Pixelating image '{input_image_path}' with factor {downscale_factor}...")
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    if downscale_factor <= 0:
        print(f"Job {job_id}: Invalid pixelate factor {downscale_factor}, defaulting to 1 (no change).")
        downscale_factor = 1

    try:
        def _pixelate_sync():
            img = Image.open(input_image_path)
            original_width, original_height = img.size
            
            # Ensure final dimensions are at least 1x1
            final_width = max(1, original_width // downscale_factor)
            final_height = max(1, original_height // downscale_factor)

            # Pixelate: Resize down with BOX, then resize up with NEAREST
            # If we want the image to stay small, we skip the upscale.
            # The request mentioned "resize" then "color quantise", implying final small pixel art.
            # If the goal is a larger image that *looks* pixelated, then upscale.
            # For now, let's assume the goal is a small pixelated image.
            pixelated_img_small = img.resize((final_width, final_height), Image.Resampling.BOX)
            
            # If you want to upscale it back to original size but pixelated:
            # pixelated_img_large = pixelated_img_small.resize((original_width, original_height), Image.Resampling.NEAREST)
            # pixelated_img_large.save(output_image_path)

            pixelated_img_small.save(output_image_path)

        await asyncio.to_thread(_pixelate_sync)
        print(f"Job {job_id}: Pixelation complete. Output: {output_image_path}")
        return True
    except Exception as e:
        print(f"Job {job_id}: Error pixelating {input_image_path}: {e}")
        return False

async def process_image_quantize_async(input_image_path, output_image_path, job_id, num_colors):
    print(f"Job {job_id}: Quantizing image '{input_image_path}' to {num_colors} colors...")
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    try:
        def _quantize_sync():
            img = Image.open(input_image_path).convert('RGBA') # Ensure RGBA for alpha handling
            
            # Preserve original alpha channel
            original_alpha = None
            if 'A' in img.getbands():
                original_alpha = img.split()[-1]

            # Quantize RGB part
            rgb_img = img.convert('RGB')
            quantized_p_img = rgb_img.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            final_rgb_img = quantized_p_img.convert("RGB") # Ensure it's RGB after quantize

            # Merge back with original alpha
            if original_alpha:
                final_output_img = Image.merge("RGBA", final_rgb_img.split() + (original_alpha,))
            else: # If no alpha, just use the RGB image
                final_output_img = final_rgb_img.convert("RGBA") # Ensure output is RGBA for consistency
            
            final_output_img.save(output_image_path)
        
        await asyncio.to_thread(_quantize_sync)
        print(f"Job {job_id}: Quantization complete. Output: {output_image_path}")
        return True
    except Exception as e:
        print(f"Job {job_id}: Error quantizing {input_image_path}: {e}")
        return False

# --- PROGRESS BAR (Simplified for single image) ---
async def update_job_status(message_to_edit, job_id, current_status_text):
    try:
        await message_to_edit.edit(content=f"Job `{job_id}`: {current_status_text}")
    except discord.NotFound:
        print(f"Job {job_id}: Progress message not found, perhaps it was deleted.")
    except Exception as e:
        print(f"Job {job_id}: Error updating status message: {e}")


# --- MAIN PROCESSING PIPELINE FOR A SINGLE IMAGE JOB ---
async def execute_image_generation_pipeline(
    ctx_message, # Original Discord message context
    job_id, 
    job_dir, 
    raw_gen_dir, 
    no_bg_dir, 
    pixelated_dir, 
    final_quantized_dir,
    params # Dictionary of parsed parameters
    ):

    status_message = await ctx_message.channel.send(f"Job `{job_id}`: Accepted. Starting image generation...")
    
    generated_image_path = os.path.join(raw_gen_dir, f"{job_id}_generated.png")
    no_bg_image_path = os.path.join(no_bg_dir, f"{job_id}_no_bg.png")
    pixelated_image_path = os.path.join(pixelated_dir, f"{job_id}_pixelated.png")
    final_image_path = os.path.join(final_quantized_dir, f"{job_id}_final.png")

    try:
        # Stage 1: Generate Image
        await update_job_status(status_message, job_id, f"🎨 Generating image with FLUX using prompt: \"{params['prompt'][:50]}...\"")
        gen_success, final_img_size = await generate_image_with_flux(
            prompt_text=params['prompt'],
            output_image_path=generated_image_path,
            job_id=job_id,
            model_name=params['model'],
            n_images=1, # Forcing 1 image
            image_size=params['size'],
            steps=params['steps'],
            guidance=params['guidance'],
            seed=params['seed'],
            t5_padding=params.get('t5_padding', DEFAULT_T5_PADDING) # Optional, fallback to default
        )
        if not gen_success: raise Exception("Image generation failed.")
        await update_job_status(status_message, job_id, f"✅ Image generated ({final_img_size[0]}x{final_img_size[1]}). Starting post-processing...")
        
        current_processing_input = generated_image_path

        # Stage 2: Background Removal (Conditional)
        if not params['skip_bg']:
            await update_job_status(status_message, job_id, "👻 Removing background...")
            bg_remove_success = await process_image_remove_background_async(current_processing_input, no_bg_image_path, job_id, params['hq_bg'])
            if not bg_remove_success:
                await ctx_message.channel.send(f"Job `{job_id}`: Warning - Background removal failed. Proceeding with original background.")
                shutil.copy(current_processing_input, no_bg_image_path) # Use original if failed
            current_processing_input = no_bg_image_path
        else:
            await update_job_status(status_message, job_id, "⏭️ Background removal skipped.")
            shutil.copy(current_processing_input, no_bg_image_path) # Copy to next stage path
            current_processing_input = no_bg_image_path


        # Stage 3: Pixelate (Resize)
        await update_job_status(status_message, job_id, f"👾 Pixelating (Factor: {params['pixelate_factor']})...")
        pixelate_success = await process_image_pixelate_async(current_processing_input, pixelated_image_path, job_id, params['pixelate_factor'])
        if not pixelate_success: raise Exception("Pixelation failed.")
        current_processing_input = pixelated_image_path

        # Stage 4: Quantize Colors
        await update_job_status(status_message, job_id, f"🎨 Quantizing colors (Colors: {params['quantize_colors']})...")
        quantize_success = await process_image_quantize_async(current_processing_input, final_image_path, job_id, params['quantize_colors'])
        if not quantize_success: raise Exception("Color quantization failed.")

        # All steps successful
        await update_job_status(status_message, job_id, "✨ Processing complete! Sending your images...")

        files_to_send = []
        if os.path.exists(final_image_path):
            files_to_send.append(discord.File(final_image_path, filename=f"{job_id}_final_quantized.png"))
        else:
            # This case should ideally be handled as an error before reaching here
            # but as a fallback, we note it.
            print(f"Job {job_id}: Final image not found at {final_image_path} for sending.")
            # raise Exception("Final image not found after processing.") # Original behavior

        if os.path.exists(generated_image_path):
            files_to_send.append(discord.File(generated_image_path, filename=f"{job_id}_generated_raw.png"))
        else:
            # This is less critical if the final image exists, but good to note.
            print(f"Job {job_id}: Raw generated image not found at {generated_image_path} for sending.")

        if files_to_send:
            await ctx_message.channel.send(files=files_to_send)
            await status_message.edit(content=f"Job `{job_id}`: Done! Here's your processed images, with the pixelified on the left, raw image on the right:")
        elif not os.path.exists(final_image_path): # Critical error if final image (main product) is missing
             raise Exception("Final image not found after processing, and no images could be sent.")
        else: # Only raw image was missing, but final was there (and sent if logic above was different)
            # This path might not be hit if final_image_path check above is strict
            await status_message.edit(content=f"Job `{job_id}`: Done! Final image sent, but raw image was missing. ✨")

    except Exception as e:
        error_msg = f"Job `{job_id}`: An error occurred: {e}. Check bot logs for details."
        print(f"ERROR in job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        try:
            await status_message.edit(content=error_msg)
        except: # If status message itself fails
             await ctx_message.channel.send(error_msg)
        await ctx_message.channel.send(f"Sorry, an error occurred while processing your request for job `{job_id}`.")
    finally:
        # Optional: Clean up job_dir after processing or keep for debugging
        # shutil.rmtree(job_dir)
        # print(f"Job {job_id}: Processing finished. Job directory '{job_dir}' {'cleaned up' if os.path.exists(job_dir) else 'retained'}.")
        print(f"Job {job_id}: Processing finished. Output (if any) is in {job_dir}")


# --- BACKGROUND TASK TO PROCESS QUEUE ---
@tasks.loop(seconds=2.0)
async def queue_processor_task():
    if not processing_queue.empty():
        print(f"Queue size: {processing_queue.qsize()}")
        (ctx_msg, job_id, j_dir, raw_dir, nobg_dir, px_dir, final_dir, parsed_params) = await processing_queue.get()
        print(f"Processing job {job_id} from queue. Prompt: \"{parsed_params['prompt'][:30]}...\"")
        try:
            await execute_image_generation_pipeline(
                ctx_msg, job_id, j_dir, raw_dir, nobg_dir, px_dir, final_dir, parsed_params
            )
        except Exception as e:
            print(f"CRITICAL unhandled exception in queue_processor_task for job {job_id}: {e}")
            await ctx_msg.channel.send(f"A critical system error occurred processing job `{job_id}`. Please contact bot admin.")
        finally:
            processing_queue.task_done()
            print(f"Finished job {job_id}. Remaining in queue: {processing_queue.qsize()}")

# --- ARGUMENT PARSING FOR BOT COMMANDS ---
def parse_generation_parameters(args_list: list, full_command_text: str):
    params = {
        'prompt': "",
        'model': DEFAULT_MODEL_NAME,
        'size': DEFAULT_IMAGE_SIZE,
        'steps': DEFAULT_STEPS,
        'guidance': DEFAULT_GUIDANCE,
        'seed': None, # Random seed by default
        'pixelate_factor': DEFAULT_PIXELATE_FACTOR,
        'quantize_colors': DEFAULT_QUANTIZE_COLORS,
        'skip_bg': DEFAULT_SKIP_BG_REMOVAL,
        'hq_bg': DEFAULT_HQ_BG_REMOVAL,
    }

    # Extract prompt: everything not a flag starting with --
    # A simple way: find first --, everything before is prompt (after command itself)
    # Or, collect non-flag parts.
    # For now, let's assume flags are parsed, and the rest is prompt.
    # We need to extract the actual command part first.
    # A more robust parser might be needed for complex prompts with "--" in them.

    # Use regex to find flags and their values
    # Example: --size 1024x768, --steps 10, --prompt "a cat"
    
    # First, try to extract a specific --prompt "text" argument
    prompt_match = re.search(r'--prompt\s+("([^"]*)"|\'([^\']*)\'|(\S+))', full_command_text, re.IGNORECASE)
    remaining_text = full_command_text
    if prompt_match:
        params['prompt'] = prompt_match.group(2) or prompt_match.group(3) or prompt_match.group(4)
        # Remove the parsed --prompt argument from the string to avoid re-parsing
        remaining_text = full_command_text[:prompt_match.start()] + full_command_text[prompt_match.end():]
    
    # Parse other flags from the remaining text
    current_args = remaining_text.split() # Split by space
    
    # Fallback prompt: if --prompt wasn't used, take the non-flag parts
    if not params['prompt']:
        non_flag_parts = []
        skip_next = False
        for i, arg in enumerate(args_list): # args_list is from original message split by space, after command
            if skip_next:
                skip_next = False
                continue
            if arg.startswith("--"):
                # Check if this flag takes a value that is also in args_list
                if arg.lower() in ["--size", "--steps", "--guidance", "--seed", "--pixelate", "--colors", "--model"]:
                    skip_next = True 
            else:
                non_flag_parts.append(arg)
        if non_flag_parts:
             params['prompt'] = " ".join(non_flag_parts)


    # Simplified flag parsing logic
    i = 0
    while i < len(current_args):
        arg = current_args[i].lower()
        if arg == "--size":
            if i + 1 < len(current_args):
                try:
                    w_str, h_str = current_args[i+1].split('x')
                    params['size'] = (int(w_str), int(h_str))
                    i += 1
                except ValueError:
                    print(f"Warning: Could not parse size '{current_args[i+1]}'. Using default.")
        elif arg == "--steps":
            if i + 1 < len(current_args):
                try: params['steps'] = int(current_args[i+1]); i += 1
                except ValueError: print(f"Warning: Invalid steps value. Using default.")
        elif arg == "--guidance":
            if i + 1 < len(current_args):
                try: params['guidance'] = float(current_args[i+1]); i += 1
                except ValueError: print(f"Warning: Invalid guidance value. Using default.")
        elif arg == "--seed":
            if i + 1 < len(current_args):
                try: params['seed'] = int(current_args[i+1]); i += 1
                except ValueError: print(f"Warning: Invalid seed value. Using random.")
        elif arg == "--model":
            if i + 1 < len(current_args) and current_args[i+1].lower() in ["schnell", "dev"]:
                params['model'] = current_args[i+1].lower()
                i += 1
            else: print(f"Warning: Invalid model. Use 'schnell' or 'dev'. Using default.")
        elif arg == "--pixelate":
            if i + 1 < len(current_args):
                try: params['pixelate_factor'] = int(current_args[i+1]); i += 1
                except ValueError: print(f"Warning: Invalid pixelate factor. Using default.")
        elif arg == "--colors":
            if i + 1 < len(current_args):
                try: params['quantize_colors'] = int(current_args[i+1]); i += 1
                except ValueError: print(f"Warning: Invalid colors value. Using default.")
        elif arg == "--no-bg":
            params['skip_bg'] = True
        elif arg == "--hq-bg":
            params['hq_bg'] = True
        # The part of the prompt might be here if not caught by --prompt
        # This simple parser might misinterpret parts of a complex prompt as flags.
        # A library like `discord.py`'s command system with converters is more robust.
        i += 1
        
    if not params['prompt']: # If still no prompt
        params['prompt'] = "default prompt, a beautiful landscape" # Fallback if really nothing
        print("Warning: No prompt provided, using a default one.")

    print(f"Parsed params: {params}")
    return params

# --- BOT EVENTS ---
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')
    print(f"Command Prefix: {COMMAND_PREFIX} (e.g. {COMMAND_PREFIX} generate <your prompt>)")
    print(f"Or mention me: @{bot.user.name} <your prompt> --size 600x600 etc.")
    print(f"Invite link: https://discord.com/api/oauth2/authorize?client_id={bot.user.id}&permissions=3072&scope=bot") # Read/Send Messages, Attach Files

    if not os.path.exists(PROCESSING_BASE_DIR):
        os.makedirs(PROCESSING_BASE_DIR)
        print(f"Created base directory for processing jobs: {PROCESSING_BASE_DIR}")
    
    check_ffmpeg() # Check ffmpeg, though not directly used by core logic

    if not queue_processor_task.is_running():
        queue_processor_task.start()
        print("Queue processor task started.")

@bot.command(name="generate", aliases=["gen", "create", "imagine"])
async def generate_command(ctx, *, full_prompt_and_args: str = ""):
    print(f"\n--- Command '{ctx.invoked_with}' received ---")
    print(f"Author: {ctx.author} (ID: {ctx.author.id})")
    print(f"Channel: {ctx.channel} (ID: {ctx.channel.id})")
    print(f"Full Input: {full_prompt_and_args}")

    if not full_prompt_and_args.strip():
        await ctx.send(f"Please provide a prompt and optionally, arguments! \nExample: `{COMMAND_PREFIX}{ctx.invoked_with} a cat wizard --steps 10 --size 768x512`")
        return

    # Use args_list for the simple parser, full_command_text for the regex --prompt
    args_list_for_parser = full_prompt_and_args.split() # This is used by the simple non --prompt part of parser
    parsed_params = parse_generation_parameters(args_list_for_parser, full_prompt_and_args)

    if not parsed_params['prompt'] or parsed_params['prompt'].isspace():
         await ctx.send("It seems you didn't provide a prompt text. Please try again, e.g., `a happy dog` or using `--prompt \"your text\"`.")
         return

    job_id = str(uuid.uuid4())
    job_dir, raw_dir, nobg_dir, px_dir, final_dir = create_job_directories(job_id)

    await ctx.send(f"Your request for \"{parsed_params['prompt'][:100]}...\" has been queued! Job ID: `{job_id}`")
    
    await processing_queue.put((
        ctx.message, # Pass the original message for context (replying, channel etc)
        job_id, job_dir, raw_dir, nobg_dir, px_dir, final_dir,
        parsed_params
    ))
    print(f"Added job {job_id} to queue. Queue size: {processing_queue.qsize()}")


@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Process commands first
    # This allows $img generate ... to work
    # It also ensures that if a command is matched, on_message doesn't re-process it if also mentioned.
    await bot.process_commands(message)
    
    # If message was processed by a command, stop here.
    # Check if context has a command associated with it. This is a bit tricky.
    # A simpler way: if message starts with prefix, assume process_commands handled it.
    # However, process_commands runs regardless. The goal is to avoid double processing if mentioned AND used command.
    # For now, if mentioned and NOT starting with command prefix, then try to parse as a mention-command.
    
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mention = bot.user.mentioned_in(message)
    # Check if message starts with any known command prefix to avoid double handling if mentioned + command
    is_command_like = False
    if message.content.startswith(COMMAND_PREFIX): # handles $img
        is_command_like = True
    # Also check for bot.command_prefix if it's a list/tuple
    if isinstance(bot.command_prefix, (list, tuple)):
        for pfx in bot.command_prefix:
            if message.content.startswith(pfx):
                is_command_like = True
                break
    
    if (is_dm or is_mention) and not is_command_like:
        print(f"\n--- Message Event (DM or Mention) ---")
        print(f"Author: {message.author} (Is Bot: {message.author.bot})")
        print(f"Channel: {message.channel} (Type: {message.channel.type})")
        print(f"Content: {message.content}")
        
        # Attempt to parse as if it's a command without the prefix
        # Remove mentions to get the "command" part
        cleaned_content = message.content
        for mention in message.mentions:
            cleaned_content = cleaned_content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "").strip()

        if not cleaned_content:
            await message.channel.send("You mentioned me! Tell me what to generate, e.g. `@MyBotName a happy cat --steps 5`")
            return

        # Now, parse this cleaned_content like it's a command
        # The first word could be an implicit "generate"
        # This is a simple way; a more robust solution would use a "default" command or more complex parsing.
        
        # Let's assume any mention implies a "generate" command
        # The `parse_generation_parameters` expects a list of args (like from .split()) and the full text
        args_list_for_parser = cleaned_content.split()
        parsed_params = parse_generation_parameters(args_list_for_parser, cleaned_content)

        if not parsed_params['prompt'] or parsed_params['prompt'].isspace():
            await message.channel.send("It seems you didn't provide a prompt text. Please try again, e.g., `@MyBotName a happy dog` or using `--prompt \"your text\"`.")
            return

        job_id = str(uuid.uuid4())
        job_dir, raw_dir, nobg_dir, px_dir, final_dir = create_job_directories(job_id)

        await message.channel.send(f"Your request for \"{parsed_params['prompt'][:100]}...\" has been queued. Job ID: `{job_id}`")
        
        await processing_queue.put((
            message, 
            job_id, job_dir, raw_dir, nobg_dir, px_dir, final_dir,
            parsed_params
        ))
        print(f"Added job {job_id} to queue. Queue size: {processing_queue.qsize()}")
    elif not is_command_like:
        # If not a DM, not a mention, and not a command, do nothing.
        pass


# --- MAIN BOT EXECUTION ---
async def main():
    if BOT_TOKEN == "YOUR_IMAGEBOT_TOKEN" or not BOT_TOKEN:
        print("ERROR: Bot token is not configured. Please set the IMAGEBOT_DISCORD_TOKEN environment variable or update the script.")
        return
    
    print("Starting ImageBot...")
    try:
        # Attempt to load FluxPipeline here to catch early errors if models aren't found
        # This is a basic check; more sophisticated health checks might be needed.
        # FluxPipeline("flux-" + DEFAULT_MODEL_NAME, t5_padding=DEFAULT_T5_PADDING).ensure_models_are_loaded()
        # print(f"Successfully pre-initialized or checked FluxPipeline with model 'flux-{DEFAULT_MODEL_NAME}'.")
        # Commented out pre-init as it might be slow for startup and better handled per job with MAX_CONCURRENT_JOBS=1
        pass
    except Exception as e:
        print(f"CRITICAL: Failed to initialize FluxPipeline during startup check: {e}")
        print("The bot may not function correctly. Ensure models are downloaded and accessible.")
        # return # Optionally, prevent startup

    async with bot:
        await bot.start(BOT_TOKEN)

if __name__ == "__main__":
    # Ensure the processing base directory exists
    if not os.path.exists(PROCESSING_BASE_DIR):
        os.makedirs(PROCESSING_BASE_DIR)
        print(f"Created base directory for processing jobs: {PROCESSING_BASE_DIR}")
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot shutting down...")
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc() 