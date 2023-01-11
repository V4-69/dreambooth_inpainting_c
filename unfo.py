seed = None # seed value for reproducible training
if seed is None or seed == 0:
    seed = random.randint(1, 999999)
else:
    seed = int(seed)

# pretrained model name or path to be used for training
pretrained_model_name_or_path = "runwayml/stable-diffusion-inpainting" 

# path to the directory containing the instance images for training
instance_data_dir = INSTANCE_DIR 

# directory for saving model predictions and checkpoints
output_dir = OUTPUT_DIR 

# directory for session data
session_dir = SESSION_DIR 

# directory for captions
captions_dir = session_dir + "/captions" 

# number of steps to train the text encoder
stop_text_encoder_training = 200 

# maximum number of training steps
all_train_steps = 3000

# save the model every n global_steps
save_n_steps = 200 

# maximum number of training steps
max_train_steps = save_n_steps 

# filename for image captions
image_captions_filename = True 

# flag to only train the U-Net
train_only_unet = True 

# starting step at which to save the model
save_starting_step = 0 

# resolution for input images
resolution = 512 

# precision level for training
mixed_precision = "fp16" 

# batch size for training
train_batch_size = 1 

# number of gradient accumulation steps
gradient_accumulation_steps = 1 

# flag to enable gradient checkpointing
gradient_checkpointing = True 

# learning rate for training
learning_rate = 1e-05 

# learning rate scheduler to use
lr_scheduler = "polynomial" 

# number of warmup steps for the learning rate scheduler
lr_warmup_steps = 0 

# prompt identifying the instance images
instance_prompt = "" 

# whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.
resume_from_checkpoint = None 



#tokenizer_name = None # tokenizer name or path to be used with the model
#class_data_dir = None # path to the directory containing the class images for training, if applicable
#class_prompt = "" # prompt identifying the class images, if applicable
#with_prior_preservation = False # flag to include prior preservation loss in training
#prior_loss_weight = 1.0 # weight of the prior preservation loss
#num_class_images = 100 # minimum number of class images for prior preservation loss
#center_crop = False # flag to center crop images before resizing to resolution

def train(pretrained_model_name_or_path, stop_text_encoder_training, max_train_steps, resume_from_checkpoint):

    !python train_dreambooth_inpaint_V2.py \
        --pretrained_model_name_or_path="{pretrained_model_name_or_path}" \
        --instance_data_dir="{instance_data_dir}" \
        --instance_prompt="{instance_prompt}" \
        --output_dir="{output_dir}" \
        --seed={seed} \
        --resolution={resolution} \
        --stop_text_encoder_training={stop_text_encoder_training} \
        --image_captions_filename={image_captions_filename} \
        --train_only_unet={train_only_unet} \
        --save_starting_step={save_starting_step} \
        --save_n_steps={save_n_steps} \
        --session_dir="{session_dir}" \
        --captions_dir="{captions_dir}" \
        --mixed_precision="{mixed_precision}" \
        --train_batch_size={train_batch_size} \
        --gradient_accumulation_steps={gradient_accumulation_steps} \
        --gradient_checkpointing={gradient_checkpointing} \
        --learning_rate={learning_rate} \
        --lr_scheduler="{lr_scheduler}" \
        --lr_warmup_steps={lr_warmup_steps} \
        --max_train_steps={max_train_steps} \
        --resume_from_checkpoint={resume_from_checkpoint}
        
    #    --tokenizer_name="{tokenizer_name}" \
    #    --class_data_dir="{class_data_dir}" \
    #    --class_prompt="{class_prompt}" \
    #    --with_prior_preservation={with_prior_preservation} \
    #    --prior_loss_weight={prior_loss_weight} \
    #    --num_class_images={num_class_images} \
    #    --center_crop={center_crop} \

def save(file_name):
    
    ckpt_path = SESSION_DIR + "/"  + Session_Name + file_name + "-16-inpainting.ckpt"

    fp16 = True
    half_arg = ''
    if fp16:
        half_arg = "--half"

    !python convert_diffusers_to_original_stable_diffusion.py --model_path $OUTPUT_DIR  --checkpoint_path $ckpt_path $half_arg
    print(f"[*] Converted ckpt saved at {ckpt_path}")


train(pretrained_model_name_or_path, stop_text_encoder_training, max_train_steps, resume_from_checkpoint)
file_name = str(max_train_steps)
save(file_name)

pretrained_model_name_or_path = output_dir
stop_text_encoder_training = 0
max_train_steps += stop_text_encoder_training
resume_from_checkpoint = "latest"