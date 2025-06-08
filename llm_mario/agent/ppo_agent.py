import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
import json
from typing import Dict, Any, List, Tuple
import config
from .agent_utils import get_model_inputs


class PolicyValueHead(nn.Module):
    """Add linear layers on top of final hidden state for policy and value networks"""
    def __init__(self, hidden_size, num_actions):
        super().__init__()

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states):
        if len(hidden_states.shape) == 3:
            llm_features = hidden_states[:, -1, :]  # get hidden state for last token
        else:
            llm_features = hidden_states
            
        action_logits = self.policy_head(llm_features)
        state_value = self.value_head(llm_features)
        
        return action_logits, state_value.squeeze(-1)


class PPOAgent:
    def __init__(self, model_path: str = None):
        self.device = config.DEVICE
        self.model_path = model_path or config.MODEL_PATH
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
            
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            quantization_config=config.QUANT_CONFIG
        )
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
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

        self.model.print_trainable_parameters()
        
        self.policy_value_head = PolicyValueHead(
            hidden_size=self.model.config.hidden_size,
            num_actions=len(config.ACTION_MAPPING)
        ).to(self.device).to(torch.bfloat16)
        
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.policy_value_head.parameters()),
            lr=config.PPO_LEARNING_RATE
        )
        
        self.experience_buffer = []
        self.current_episode_data = []

    def choose_action(self, screenshot: str, game_state: Dict[str, Any]) -> Tuple[int, float, float]:
        self.model.eval()
        self.policy_value_head.eval()
        
        # Clear cache before action selection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            exp = {
                'game_state': game_state,
                'screenshot': screenshot
            }
            inputs = get_model_inputs(exp, self.processor, config.MAX_LENGTH).to(self.device)
            
            # Get LLM hidden states with memory optimization
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
                hidden_states = outputs.hidden_states[-1]  # Last layer
                
                # Get action probabilities and value
                action_logits, state_value = self.policy_value_head(hidden_states)
                
                # Sample action
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = Categorical(action_probs)
                action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action)
                
                result = (action.item(), action_log_prob.item(), state_value.item())
            
            # Clean up memory
            del inputs, outputs, hidden_states, action_logits, state_value
            del action_probs, action_dist, action, action_log_prob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return result

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
        
        print(f"Training PPO for {config.PPO_EPOCHS} epochs")
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
        """Train on a single minibatch with memory optimization"""
        # Extract data with consistent dtypes
        actions = torch.tensor([exp['action'] for exp in minibatch], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([exp['log_prob'] for exp in minibatch], dtype=torch.bfloat16).to(self.device)
        advantages = torch.tensor([exp['advantage'] for exp in minibatch], dtype=torch.bfloat16).to(self.device)
        returns = torch.tensor([exp['return'] for exp in minibatch], dtype=torch.bfloat16).to(self.device)
        
        # Use gradient accumulation to reduce memory usage
        self.optimizer.zero_grad()
        total_loss = 0.0
        accumulation_steps = max(1, len(minibatch) // 2)  # Process 2 samples at a time
        
        action_logits_list = []
        values_list = []
        
        # Process samples in smaller chunks to manage memory
        for i in range(0, len(minibatch), accumulation_steps):
            chunk_end = min(i + accumulation_steps, len(minibatch))
            chunk = minibatch[i:chunk_end]
            chunk_actions = actions[i:chunk_end]
            
            chunk_logits = []
            chunk_values = []
            
            # Process each sample in chunk
            for exp in chunk:
                state_data = exp['state_data']
                
                # Clear cache before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                inputs = get_model_inputs(state_data, self.processor, config.MAX_LENGTH).to(self.device)
                
                # Get LLM hidden states with memory optimization
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
                    hidden_states = outputs.hidden_states[-1]
                    
                    action_logits, state_value = self.policy_value_head(hidden_states)
                    chunk_logits.append(action_logits)
                    chunk_values.append(state_value)
                
                # Clear intermediate variables
                del inputs, outputs, hidden_states
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Stack chunk tensors
            chunk_action_logits = torch.stack(chunk_logits)
            chunk_values_tensor = torch.stack(chunk_values)
            
            action_logits_list.extend(chunk_logits)
            values_list.extend(chunk_values)
            
            del chunk_logits, chunk_values, chunk_action_logits, chunk_values_tensor
            
        # Stack all tensors
        all_action_logits = torch.stack(action_logits_list)
        all_values = torch.stack(values_list).squeeze(-1)
        
        # Calculate losses
        action_probs = F.softmax(all_action_logits, dim=-1)
        action_dist = Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        
        # PPO policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.PPO_CLIP_COEF, 1 + config.PPO_CLIP_COEF) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss - ensure consistent shapes
        value_loss = F.mse_loss(all_values, returns)
        
        # Entropy loss
        entropy_loss = -action_dist.entropy().mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     config.PPO_VALUE_COEF * value_loss + 
                     config.PPO_ENTROPY_COEF * entropy_loss)
        
        # Backward pass with gradient scaling
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.policy_value_head.parameters()),
            config.MAX_GRAD_NORM
        )
        self.optimizer.step()
        
        # Clean up
        del all_action_logits, all_values, action_probs, action_dist
        del policy_loss, value_loss, entropy_loss, ratio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return total_loss.item()

    def save_model(self, path):
        self.model.save_pretrained(path)
        torch.save(self.policy_value_head.state_dict(), f"{path}/policy_value_head.pt")
        
    def load_model(self, path):
        self.model.load_adapter(path)
        self.policy_value_head.load_state_dict(torch.load(f"{path}/policy_value_head.pt"))

    def reset_episode(self):
        if self.current_episode_data:
            self.calculate_advantages()
            self.experience_buffer.extend(self.current_episode_data)
            self.current_episode_data = [] 