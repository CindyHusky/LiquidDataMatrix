import cv2
import numpy as np
import time
import torch
from pynput.keyboard import Controller, Key

# === PARAMETERS ===
LEARNING_SPACE_SIZE = (2048, 2048)  # New larger learning space
BATCH_SIZE = 16                      # Number of frames processed in parallel
DECAY_LEARNING = 0.98                # Faster fading for better learning
BASE_BLEND = 0.90                    # Base weight for learning space updates
NOVEL_BLEND = 0.10                   # Extra weight for novel regions
DIFF_THRESHOLD = 30                  # Pixel difference threshold for novelty

# Novelty/reward parameters
PREDICTION_THRESHOLD = 0.5           # Threshold for histogram difference to be considered novel
CURIOSITY_FACTOR = 0.1               # Multiplier for the novelty to compute reward

# Long-term memory parameters
DECAY_FACTOR = 0.999                 # Slower decay for long-term relevance
MAX_MEMORY = 10_000                  # Store up to 10,000 learning states

# GPU Utilization
USE_GPU = torch.cuda.is_available()  # Enable GPU if available
DEVICE = "cuda" if USE_GPU else "cpu" # Select device

# === INITIAL SETUP ===
keyboard = Controller()
learning_space = torch.zeros((BATCH_SIZE, *LEARNING_SPACE_SIZE), dtype=torch.float32, device=DEVICE)
long_term_memory = []

# === HELPER FUNCTIONS ===

def batch_histogram_novelty(batch_frames, memory_frames):
    """Compute histogram difference for a batch of frames."""
    batch_frames = batch_frames.to(DEVICE)
    hist_batch = torch.histc(batch_frames, bins=256, min=0, max=255).float()
    hist_batch /= hist_batch.sum()

    novelty_scores = []
    for mem_frame in memory_frames:
        hist_mem = torch.histc(mem_frame.to(DEVICE), bins=256, min=0, max=255).float()
        hist_mem /= hist_mem.sum()
        diff = torch.norm(hist_batch - hist_mem, p=1)  # L1 Norm
        novelty_scores.append(diff.item())

    return sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0

def calculate_curiosity_reward(batch_frames, memory_frames):
    """Compute curiosity reward using batch novelty detection."""
    novelty = batch_histogram_novelty(batch_frames, memory_frames)
    reward = novelty * CURIOSITY_FACTOR if novelty > PREDICTION_THRESHOLD else 0
    return reward, novelty

def batch_update_learning_space(learning_space, batch_frames):
    """Efficient batch update for learning space using parallel computation."""
    batch_frames = batch_frames.to(DEVICE)
    decayed = learning_space * DECAY_LEARNING
    diff = torch.abs(batch_frames - decayed)
    mask = (diff > DIFF_THRESHOLD).float()
    updated = decayed * BASE_BLEND + batch_frames * (1 - BASE_BLEND + mask * NOVEL_BLEND)
    return torch.nn.functional.avg_pool2d(updated, kernel_size=3, stride=1, padding=1)

def update_long_term_memory(memory, batch_frames):
    """Batch update for long-term memory (stores up to 10,000 frames)."""
    batch_frames = batch_frames.to(DEVICE)

    if len(memory) < MAX_MEMORY:
        memory.extend(batch_frames)
        print(f"Memory added. Total: {len(memory)}.")
    else:
        # Compute batch novelty
        batch_novelty = torch.mean(torch.stack([torch.norm(batch_frame - state, p=1) for batch_frame in batch_frames for state in memory]))

        # Compute importance scores for memory
        importance_scores = []
        for state in memory:
            differences = [torch.norm(state - other, p=1) for other in memory if not torch.equal(state, other)]
            importance = torch.mean(torch.stack(differences)) if differences else torch.mean(state)
            importance_scores.append(importance)

        min_index = torch.argmin(torch.tensor(importance_scores))
        min_importance = importance_scores[min_index]

        if batch_novelty > min_importance:
            memory[min_index] = batch_frames[0]  # Replace least important memory with new frame
            print(f"Replaced memory at index {min_index}: New novelty {batch_novelty:.2f} > old {min_importance:.2f}.")
        else:
            print(f"New batch novelty ({batch_novelty:.2f}) did not exceed minimum importance ({min_importance:.2f}). No replacement.")

    return memory

def decide_action(learning_space, curiosity_reward):
    """Determine an action using brightness and curiosity."""
    avg_brightness = torch.mean(learning_space).item()
    
    if curiosity_reward > 0.2:
        print("Curiosity triggered movement.")
        return np.random.choice([0, 1, 2, 3])  # Random movement
    elif avg_brightness > 0.5:
        return 0  # Move forward
    elif avg_brightness < 0.2:
        return 3  # Jump
    else:
        return np.random.choice([1, 2])  # Turn left or right

# === MAIN LOOP ===

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit(1)

frame_count = 0
batch_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1
    frame_resized = cv2.resize(frame, LEARNING_SPACE_SIZE, interpolation=cv2.INTER_AREA)
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    batch_frames.append(gray_frame)
    if len(batch_frames) < BATCH_SIZE:
        continue  # Wait until we have a full batch

    batch_tensor = torch.stack([torch.tensor(f, dtype=torch.float32, device=DEVICE) for f in batch_frames])

    # Update learning space in parallel for all batch frames
    learning_space = batch_update_learning_space(learning_space, batch_tensor)

    # Compute curiosity reward in parallel
    curiosity_reward, raw_novelty = calculate_curiosity_reward(batch_tensor, [state for state in long_term_memory])
    if curiosity_reward > 0:
        print(f"[Frame {frame_count}] Novelty reward: {curiosity_reward:.3f} (raw: {raw_novelty:.3f})")

    # Update long-term memory in parallel
    long_term_memory = update_long_term_memory(long_term_memory, batch_tensor)

    # Determine and execute action
    action = decide_action(learning_space, curiosity_reward)
    perform_action(action)
    print(f"Performed action: {action}")

    # Ensure valid image type for OpenCV (2D, uint8)
    try:
        learning_space_np = learning_space.cpu().numpy()

        # Make sure the tensor is 2D and in uint8 format for OpenCV to handle it
        if learning_space_np.ndim == 2:  # If it's already 2D (grayscale)
            learning_space_np = np.uint8(learning_space_np)
        elif learning_space_np.ndim == 3:  # In case it's 3D (RGB, might need further processing)
            learning_space_np = np.uint8(learning_space_np[0, :, :])  # Just select one channel

        if learning_space_np.size > 0:
            cv2.imshow('Learning Space', learning_space_np)
        else:
            print("Error: Learning space is empty, skipping display.")
    except Exception as e:
        print(f"Error displaying learning space: {e}")

    try:
        cv2.imshow('Current Frame', gray_frame)
    except Exception as e:
        print(f"Error displaying current frame: {e}")

    batch_frames = []  # Reset batch
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
