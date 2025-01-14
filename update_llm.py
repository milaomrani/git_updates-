import os
import random
import subprocess
from datetime import datetime
from transformers import pipeline
import random


def create_daily_pyfile(new_number):
    topics = ["CNN", "RNN", "LSTM", "Transformer", "GAN", "Autoencoder"]
    chosen_topic = random.choice(topics)
    code_snippet = f"""# Deep Learning Example: {chosen_topic}
import torch

print("Generating a {chosen_topic} model for number {new_number}")
# Placeholder code for {chosen_topic}:

class SimpleModel:
    def __init__(self):
        print("Initialize {chosen_topic} model")

if __name__ == "__main__":
    model = SimpleModel()
"""
    torch_layers = ["Linear", "Conv2d", "LSTM", "Transformer", "GRU"]
    layer = random.choice(torch_layers)

    if chosen_topic == "CNN":
        code_snippet += f"""
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = CNN()
print(f"Model architecture:\\n{model}")"""

    elif chosen_topic == "RNN":
        code_snippet += """
class RNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=10):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = RNN()
print(f"RNN model created with {model.hidden_size} hidden units")"""

    elif chosen_topic == "LSTM":
        code_snippet += """
class LSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTM()
print("LSTM model initialized with 2 layers")"""

    elif chosen_topic == "Transformer":
        code_snippet += """
class TransformerModel(nn.Module):
    def __init__(self, ntoken=100, d_model=200, nhead=2, nhid=200):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        
    def forward(self, src, tgt):
        src = self.encoder(src) * math.sqrt(self.d_model)
        tgt = self.encoder(tgt) * math.sqrt(self.d_model)
        output = self.transformer(src, tgt)
        return output

model = TransformerModel()
print("Transformer model created with 2 attention heads")"""

    elif chosen_topic == "GAN":
        code_snippet += """
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

generator = Generator(100)
discriminator = Discriminator()
print("GAN models initialized")"""

    elif chosen_topic == "Autoencoder":
        code_snippet += """
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
print("Autoencoder architecture created")"""
    # Additional advanced topics and their implementations
    topics.extend(["Image Segmentation", "Object Detection", "Time Series Forecasting", 
                  "Attention Mechanism", "VAE", "Self-Supervised Learning"])
    
    if chosen_topic == "Image Segmentation":
        code_snippet += """
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        x1 = torch.relu(self.enc1(x))
        x2 = torch.relu(self.enc2(x1))
        x3 = self.dec1(x2)
        return torch.sigmoid(self.dec2(x3))

model = UNet()
print("U-Net architecture for image segmentation created")"""

    elif chosen_topic == "Time Series Forecasting":
        code_snippet += """
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_heads=4):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=2
        )
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.predictor(x)

model = TimeSeriesTransformer()
print("Transformer model for time series forecasting initialized")"""

    elif chosen_topic == "Attention Mechanism":
        code_snippet += """
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size=128):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        return self.norm(x + self.mlp(x))

model = AttentionLayer()
print("Multi-head attention layer with MLP created")"""
    filename = f"deep_learning_example_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    with open(filename, "w") as f:
        f.write(code_snippet)
    return filename

def read_number():
    with open('number.txt', 'r') as f:
        return int(f.read().strip())


def initialize_files():
    if not os.path.exists('number.txt'):
        with open('number.txt', 'w') as f:
            f.write('0')
        with open('thought.txt', 'w') as f:
            f.write('Initial thought')
        subprocess.run(['git', 'add', 'number.txt', 'thought.txt'])
        subprocess.run(['git', 'commit', '-m', 'Initialize tracking'])


def generate_text(number):
    generator = pipeline('text-generation', model='gpt2')
    
    # List of programming and AI concepts
    topics = [
        "algorithms", "machine learning", "data structures",
        "neural networks", "software design", "coding patterns",
        "artificial intelligence", "deep learning"
    ]
    
    # Select random topic for variety
    topic = random.choice(topics)
    
    prompts = [
        f"Here's a coding challenge #{number}: Implement a {topic} solution that",
        f"Let's explore how {topic} relates to problem #{number} in software development:",
        f"AI concept #{number}: Understanding {topic} and its applications in",
    ]
    
    prompt = random.choice(prompts)
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

def write_files(number, text):
    with open('number.txt', 'w') as f:
        f.write(str(number))
    with open('thought.txt', 'w') as f:
        f.write(text)

def git_commit():
    os.system("git add .")
    os.system('git commit -m "Auto-commit generated code"')

def git_push():
    os.system("git push origin main")
        
def update_cron_with_random_time():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    random_hour = random.randint(0, 23)
    random_minute = random.randint(0, 59)
    new_cron_command = f"{random_minute} {random_hour} * * * cd {script_dir} && python3 {os.path.join(script_dir, 'update_number.py')}\n"
    cron_file = "/tmp/current_cron"
    os.system(f"crontab -l > {cron_file} 2>/dev/null || true")
    with open(cron_file, "r") as file:
        lines = file.readlines()
    with open(cron_file, "w") as file:
        for line in lines:
            if "update_number.py" not in line:
                file.write(line)
        file.write(new_cron_command)
    os.system(f"crontab {cron_file}")

def generate_daily_schedule(num_daily_runs=10):
    schedule = []
    
    # 9 AM to 5 PM
    initial_hours = list(range(9, 17))
    
    for _ in range(num_daily_runs):
        # Reset available hours if needed
        if not initial_hours:
            initial_hours = list(range(9, 17))
            
        if initial_hours:
            hour = random.choice(initial_hours)
            minute = random.randint(0, 59)
            schedule.append((hour, minute))
            
            # Remove selected hour and adjacent hours
            initial_hours = [h for h in initial_hours 
                           if abs(h - hour) >= 2]
    
    # Sort by time
    schedule.sort()
    return schedule

def update_cron_with_random_times():
    try:
        schedule = generate_daily_schedule()
        cron_file = "/tmp/crontab.tmp"
        script_path = os.path.abspath(__file__)
        
        # Update crontab
        os.system(f"crontab -l > {cron_file}")
        with open(cron_file, "w") as file:
            for hour, minute in schedule:
                file.write(f"{minute} {hour} * * * cd {os.path.dirname(script_path)} && python3 {script_path}\n")
        
        os.system(f"crontab {cron_file}")
        os.remove(cron_file)
        
        print(f"Updated schedule with {len(schedule)} runs")
        for hour, minute in schedule:
            print(f"- {hour:02d}:{minute:02d}")
            
    except Exception as e:
        print(f"Error updating cron: {str(e)}")
        
def create_conda_script():
    conda_script = """#!/bin/bash
source /Users/miladomrani/opt/anaconda3/etc/profile.d/conda.sh
conda activate AG
cd /Users/miladomrani/Documents/code/git_increament/git_updates-
python update_llm.py
"""
    with open('conda.sh', 'w') as f:
        f.write(conda_script)
    os.chmod('conda.sh', 0o755)

def create_run_script():
    run_script = """#!/bin/bash
/Users/miladomrani/Documents/code/git_increament/git_updates-/conda.sh
"""
    with open('run_update.sh', 'w') as f:
        f.write(run_script)
    os.chmod('run_update.sh', 0o755)

def setup_cron():
    script_path = os.path.abspath('run_update.sh')
    os.system(f'(crontab -l 2>/dev/null; echo "0 * * * * {script_path}") | crontab -')

def setup_automation():
    create_conda_script()
    create_run_script()
    setup_cron()
    print("Automation setup complete")
    

def main():
    try:
        initialize_files()
        current_number = read_number()
        new_number = current_number + 1
        generated_text = generate_text(new_number)
        write_files(new_number, generated_text)
        
        # Create a new Python file with a deep learning example
        new_file = create_daily_pyfile(new_number)
        git_commit()
        git_push()
        # update_cron_with_random_time()
        update_cron_with_random_times()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

# ...existing code...
def schedule_daily_runs(NUM_DAILY_RUNS=10):
    schedule_today = []
    LAST_HOUR = None
    
    while len(schedule_today) < NUM_DAILY_RUNS:
        business_hours = list(range(9, 17))  # Reset business hours
        if LAST_HOUR is not None:
            # Ensure DELAY hours between runs (1-2 hours)
            delay = random.randint(1, 2)
            business_hours = [h for h in business_hours if h > LAST_HOUR + delay]
        
        if not business_hours:
            break
            
        hour = random.choice(business_hours)
        schedule_today.append((hour, random.randint(0, 59)))
        LAST_HOUR = hour
    
    return schedule_today

if __name__ == "__main__":
    num_daily_runs = 10  # Set fixed number of runs per day
    setup_automation()
    
    # Generate schedule for today and tomorrow with exactly 10 runs per day
    schedule_today = schedule_daily_runs(num_daily_runs)
    
    # Sort the schedule by time
    schedule_today.sort()
    
    # Update cron with new schedule
    update_cron_with_random_times()
    main()