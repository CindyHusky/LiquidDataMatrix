#i forgor i was adapting the visual model for minecraft gameplay heres a less danegerous one lol


import cv2
import numpy as np
import time
import torch

# === PARAMETERS ===
LEARNING_SPACE_SIZE = (2048, 2048)  # New larger learning space
BATCH_SIZE = 16                     # Number of frames processed in parallel
DECAY_LEARNING = 0.98               # Faster fading for better learning
BASE_BLEND = 0.90                   # Base weight for learning space updates
NOVEL_BLEND = 0.10                  # Extra weight for novel regions
DIFF_THRESHOLD = 30                 # Pixel difference threshold for novelty

# Novelty/reward parameters
PREDICTION_THRESHOLD = 0.5          # Threshold for histogram difference to be considered novel
CURIOSITY_FACTOR = 0.1              # Multiplier for the novelty to compute reward

# Long-term memory parameters
DECAY_FACTOR = 0.999                # Slower decay for long-term relevance
MAX_MEMORY = 10_000                 # Store up to 10,000 learning states

# GPU Utilization
USE_GPU = torch.cuda.is_available()  # Enable GPU if available
DEVICE = "cuda" if USE_GPU else "cpu"  # Select device

# === INITIAL SETUP ===
learning_space = torch.zeros((BATCH_SIZE, *LEARNING_SPACE_SIZE), dtype=torch.float32, device=DEVICE)
long_term_memory = []

# === HELPER FUNCTIONS ===

def batch_histogram_novelty(batch_frames, memory_frames):
    """Compute histogram difference for a batch of frames."""
    # For simplicity we compute histogram on the first frame of the batch
    batch_frame = batch_frames[0].to(DEVICE)
    hist_batch = torch.histc(batch_frame, bins=256, min=0, max=255).float()
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
    
    # Ensure tensor shape is compatible with avg_pool2d (expects 4D: (batch, channel, H, W))
    if updated.ndim == 3:
        updated = updated.unsqueeze(1)  # now (BATCH_SIZE, 1, H, W)
        updated = torch.nn.functional.avg_pool2d(updated, kernel_size=3, stride=1, padding=1)
        updated = updated.squeeze(1)
    else:
        updated = torch.nn.functional.avg_pool2d(updated, kernel_size=3, stride=1, padding=1)
    return updated

def update_long_term_memory(memory, batch_frames):
    """Batch update for long-term memory (stores up to MAX_MEMORY frames)."""
    batch_frames = batch_frames.to(DEVICE)

    if len(memory) < MAX_MEMORY:
        memory.extend(batch_frames)
        print(f"Memory added. Total: {len(memory)}.")
    else:
        # Compute batch novelty across all memory states
        all_diffs = []
        for batch_frame in batch_frames:
            for state in memory:
                all_diffs.append(torch.norm(batch_frame - state, p=1))
        if all_diffs:
            batch_novelty = torch.mean(torch.stack(all_diffs))
        else:
            batch_novelty = torch.tensor(0.0, device=DEVICE)
    
        # Compute importance scores for memory
        importance_scores = []
        for i, state in enumerate(memory):
            differences = [torch.norm(state - other, p=1) for j, other in enumerate(memory) if i != j]
            if differences:
                importance = torch.mean(torch.stack(differences))
            else:
                importance = torch.mean(state)
            importance_scores.append(importance)

        # Find the memory state with the minimum importance
        importance_tensor = torch.tensor([score.item() for score in importance_scores], device=DEVICE)
        min_index = int(torch.argmin(importance_tensor).item())
        min_importance = importance_scores[min_index]

        if batch_novelty > min_importance:
            memory[min_index] = batch_frames[0]
            print(f"Replaced memory at index {min_index}: New novelty {batch_novelty:.2f} > old {min_importance:.2f}.")
        else:
            print(f"New batch novelty ({batch_novelty:.2f}) did not exceed minimum importance ({min_importance:.2f}). No replacement.")

    return memory

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

    # Update learning space using the batch
    learning_space = batch_update_learning_space(learning_space, batch_tensor)

    # Compute curiosity reward
    curiosity_reward, raw_novelty = calculate_curiosity_reward(batch_tensor, long_term_memory)
    if curiosity_reward > 0:
        print(f"[Frame {frame_count}] Novelty reward: {curiosity_reward:.3f} (raw: {raw_novelty:.3f})")

    # Update long-term memory with the batch frames
    long_term_memory = update_long_term_memory(long_term_memory, batch_tensor)

    # Display the learning space
    try:
        learning_space_np = learning_space.cpu().numpy()
        if learning_space_np.ndim == 2:
            learning_space_np = np.uint8(learning_space_np)
        elif learning_space_np.ndim == 3:
            learning_space_np = np.uint8(learning_space_np[0])
        if learning_space_np.size > 0:
            cv2.imshow('Learning Space', learning_space_np)
        else:
            print("Error: Learning space is empty, skipping display.")
    except Exception as e:
        print(f"Error displaying learning space: {e}")

    # Display the current frame
    try:
        cv2.imshow('Current Frame', gray_frame)
    except Exception as e:
        print(f"Error displaying current frame: {e}")

    batch_frames = []  # Reset batch for the next iteration

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
