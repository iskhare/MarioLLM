import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import json
import base64
from PIL import Image
from io import BytesIO
from typing import Dict, Any, List, Tuple
import config


class VisionEncoder(nn.Module):
    """Simple CNN to encode game screenshots"""
    def __init__(self, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, output_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))


class PolicyValueHead(nn.Module):
    """Policy and value head for the LLM"""
    def __init__(self, hidden_size, num_actions, vision_dim=256):
        super().__init__()
        self.vision_encoder = VisionEncoder(vision_dim)
        
        # Combine LLM hidden states with vision features
        self.fusion_layer = nn.Linear(hidden_size + vision_dim, hidden_size)
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Value head (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states, vision_input):
        # Encode vision input
        vision_features = self.vision_encoder(vision_input)
        
        # Take the last hidden state from LLM
        if len(hidden_states.shape) == 3:
            llm_features = hidden_states[:, -1, :]  # [batch, hidden_size]
        else:
            llm_features = hidden_states
            
        # Fuse vision and language features
        combined_features = torch.cat([llm_features, vision_features], dim=-1)
        fused_features = F.relu(self.fusion_layer(combined_features))
        
        # Get policy and value
        action_logits = self.policy_head(fused_features)
        state_value = self.value_head(fused_features)
        
        return action_logits, state_value.squeeze(-1)


class PPOAgent:
    def __init__(self, model_path: str = None):
        self.device = config.DEVICE
        self.model_path = model_path or config.LOCAL_MODEL_PATH
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)
        
        # Add policy and value heads
        hidden_size = self.model.config.hidden_size
        self.policy_value_head = PolicyValueHead(
            hidden_size=hidden_size,
            num_actions=len(config.ACTION_MAPPING)
        ).to(self.device)
        
        # PPO optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.policy_value_head.parameters()),
            lr=config.PPO_LEARNING_RATE
        )
        
        # Experience buffer
        self.experience_buffer = []
        self.current_episode_data = []
        
    def get_system_prompt(self) -> str:
        """System prompt for Mario gameplay"""
        return """You are an AI agent playing Super Mario Bros. Analyze the game state and choose the best action.

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
- Avoid getting stuck"""

    def preprocess_image(self, image_b64: str) -> torch.Tensor:
        """Convert base64 image to tensor for vision encoder"""
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            image = image.resize((84, 84))
            
            # Convert to tensor and normalize
            image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1) / 255.0
            return image_tensor.unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            # Return dummy tensor
            return torch.zeros(1, 3, 84, 84).to(self.device)

    def format_game_state(self, game_state: Dict[str, Any]) -> str:
        """Format game state for the prompt"""
        return f"""Current State:
Position: ({game_state['x_pos']}, {game_state['y_pos']})
Score: {game_state['score']} | Coins: {game_state['coins']} | Lives: {game_state['life']}
Time: {game_state['time']} | World: {game_state['world']}-{game_state['stage']}"""

    def choose_action(self, screenshot: str, game_state: Dict[str, Any]) -> Tuple[int, float, float]:
        """Choose action using the policy network"""
        self.model.eval()
        self.policy_value_head.eval()
        
        with torch.no_grad():
            # Prepare text input
            system_prompt = self.get_system_prompt()
            game_info = self.format_game_state(game_state)
            full_prompt = f"{system_prompt}\n\n{game_info}\n\nChoose action:"
            
            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                max_length=config.MAX_LENGTH,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get LLM hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Preprocess image
            vision_input = self.preprocess_image(screenshot)
            
            # Get action probabilities and value
            action_logits, state_value = self.policy_value_head(hidden_states, vision_input)
            
            # Sample action
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            
            return action.item(), action_log_prob.item(), state_value.item()

    def store_experience(self, state_data: Dict, action: int, log_prob: float, 
                        value: float, reward: float, done: bool):
        """Store experience for PPO training"""
        experience = {
            'state_data': state_data,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'done': done
        }
        self.current_episode_data.append(experience)
        
        if done:
            # Calculate advantages and returns
            self.calculate_advantages()
            self.experience_buffer.extend(self.current_episode_data)
            self.current_episode_data = []
            
            # Keep buffer size manageable
            if len(self.experience_buffer) > config.REPLAY_BUFFER_SIZE:
                self.experience_buffer = self.experience_buffer[-config.REPLAY_BUFFER_SIZE:]

    def calculate_advantages(self):
        """Calculate GAE advantages for the current episode"""
        if not self.current_episode_data:
            return
            
        rewards = [exp['reward'] for exp in self.current_episode_data]
        values = [exp['value'] for exp in self.current_episode_data]
        
        # Calculate returns and advantages using GAE
        advantages = []
        returns = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + config.PPO_GAMMA * next_value - values[i]
            gae = delta + config.PPO_GAMMA * config.PPO_GAE_LAMBDA * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store in experience data
        for i, exp in enumerate(self.current_episode_data):
            exp['advantage'] = advantages[i]
            exp['return'] = returns[i]

    def train_ppo(self):
        """Train using PPO algorithm"""
        if len(self.experience_buffer) < config.PPO_BATCH_SIZE:
            return None
            
        self.model.train()
        self.policy_value_head.train()
        
        # Sample batch
        batch_indices = np.random.choice(
            len(self.experience_buffer), 
            size=config.PPO_BATCH_SIZE, 
            replace=False
        )
        batch_data = [self.experience_buffer[i] for i in batch_indices]
        
        total_loss = 0
        
        for epoch in range(config.PPO_EPOCHS):
            # Process minibatches
            for i in range(0, len(batch_data), config.PPO_MINIBATCH_SIZE):
                minibatch = batch_data[i:i+config.PPO_MINIBATCH_SIZE]
                if len(minibatch) < config.PPO_MINIBATCH_SIZE:
                    continue
                    
                loss = self.train_minibatch(minibatch)
                total_loss += loss
                
        return total_loss / (config.PPO_EPOCHS * (len(batch_data) // config.PPO_MINIBATCH_SIZE))

    def train_minibatch(self, minibatch: List[Dict]) -> float:
        """Train on a single minibatch"""
        # Extract data
        actions = torch.tensor([exp['action'] for exp in minibatch]).to(self.device)
        old_log_probs = torch.tensor([exp['log_prob'] for exp in minibatch]).to(self.device)
        advantages = torch.tensor([exp['advantage'] for exp in minibatch]).to(self.device)
        returns = torch.tensor([exp['return'] for exp in minibatch]).to(self.device)
        
        # Process states
        action_logits_list = []
        values_list = []
        
        for exp in minibatch:
            state_data = exp['state_data']
            
            # Prepare inputs
            system_prompt = self.get_system_prompt()
            game_info = self.format_game_state(state_data['game_state'])
            full_prompt = f"{system_prompt}\n\n{game_info}\n\nChoose action:"
            
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                max_length=config.MAX_LENGTH,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get LLM hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            # Process image
            vision_input = self.preprocess_image(state_data['screenshot'])
            
            # Get predictions
            action_logits, state_value = self.policy_value_head(hidden_states, vision_input)
            action_logits_list.append(action_logits)
            values_list.append(state_value)
        
        # Stack tensors
        all_action_logits = torch.stack(action_logits_list)
        all_values = torch.stack(values_list)
        
        # Calculate losses
        action_probs = F.softmax(all_action_logits, dim=-1)
        action_dist = Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        
        # PPO policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.PPO_CLIP_COEF, 1 + config.PPO_CLIP_COEF) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(all_values, returns)
        
        # Entropy loss
        entropy_loss = -action_dist.entropy().mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     config.PPO_VALUE_COEF * value_loss + 
                     config.PPO_ENTROPY_COEF * entropy_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.policy_value_head.parameters()),
            config.PPO_MAX_GRAD_NORM
        )
        self.optimizer.step()
        
        return total_loss.item()

    def save_model(self, path: str):
        """Save the model and policy head"""
        self.model.save_pretrained(path)
        torch.save(self.policy_value_head.state_dict(), f"{path}/policy_value_head.pt")
        
    def load_model(self, path: str):
        """Load the model and policy head"""
        self.model.load_adapter(path)
        self.policy_value_head.load_state_dict(torch.load(f"{path}/policy_value_head.pt"))

    def reset_episode(self):
        """Reset episode-specific data"""
        if self.current_episode_data:
            self.calculate_advantages()
            self.experience_buffer.extend(self.current_episode_data)
            self.current_episode_data = [] 