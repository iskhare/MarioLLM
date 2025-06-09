import base64
from io import BytesIO
from PIL import Image

SYSTEM_PROMPT = """You are an AI agent playing Super Mario Bros. Analyze the game state and choose the best action.

Available actions:
0: NOOP - Do nothing
1: RIGHT - Move right
2: RIGHT+JUMP - Move right and jump
3: RIGHT+RUN - Move right and run
4: RIGHT+RUN+JUMP - Move right, run and jump (best for long jumps)
5: JUMP - Jump in place
6: LEFT - Move left (avoid unless necessary)

Strategy:
- Always progress right to complete the level
- Jump over enemies and gaps
- Use running for speed and longer jumps
- Collect coins when safe
- Avoid getting stuck
- Complete the level as fast as possible"""


def preprocess_image(image_b64: str) -> Image.Image:
    """Convert base64 image to PIL Image so it can be passed into multimodal model"""
    image_data = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    return image


def get_model_inputs(state_data, processor, max_length):
    """Turn the game state into a model-readable format"""
    game_state = state_data['game_state']
    game_info = f"""Current State:
Position: ({game_state['x_pos']}, {game_state['y_pos']})
Score: {game_state['score']} | Coins: {game_state['coins']} | Lives: {game_state['life']}
Time: {game_state['time']} | World: {game_state['world']}-{game_state['stage']}"""
    
    full_prompt = f"{SYSTEM_PROMPT}\n\n{game_info}\n\nChoose action:"
    
    # Preprocess image
    image = preprocess_image(state_data['screenshot'])
    
    # Use Qwen2.5-VL chat template format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_prompt}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return processor(
        text=text,
        images=image,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    )


def get_model_inputs_batch(state_data_list, processor, max_length):
    """Turn multiple game states into a batched model-readable format"""
    texts = []
    images = []
    
    for state_data in state_data_list:
        game_state = state_data['game_state']
        game_info = f"""Current State:
Position: ({game_state['x_pos']}, {game_state['y_pos']})
Score: {game_state['score']} | Coins: {game_state['coins']} | Lives: {game_state['life']}
Time: {game_state['time']} | World: {game_state['world']}-{game_state['stage']}"""
        
        full_prompt = f"{SYSTEM_PROMPT}\n\n{game_info}\n\nChoose action:"
        
        # Preprocess image
        image = preprocess_image(state_data['screenshot'])
        images.append(image)
        
        # Use Qwen2.5-VL chat template format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": full_prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
    
    # Process all inputs as a batch
    return processor(
        text=texts,
        images=images,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    )