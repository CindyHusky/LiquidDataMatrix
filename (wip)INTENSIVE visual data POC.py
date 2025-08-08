#updates: 
#long term actually does stuff now
#applyed smoothing to long term to allow for more perceptron data bleed
#Memorys now decay adaptively, blend into space on recall

import cv2
import numpy as np

# === PARAMETERS ===
LEARNING_SPACE_SIZE = (512, 512)  # (width, height)
DECAY_BASE = 0.98                 # Base decay rate
RECALL_BLEND = 0.4                 # How much recalled memory influences learning space
NOVELTY_THRESHOLD = 0.5            # Min novelty score to store a memory
CURIOSITY_FACTOR = 0.05            # Curiosity reward multiplier
MAX_MEMORY = 100                   # Max stored memories
FORGETTING_RATE = 0.99             # Forget unused memories faster

# === INITIAL SETUP ===
learning_space = np.zeros(LEARNING_SPACE_SIZE, dtype=np.float32)
long_term_memory = []
memory_strengths = []  # Tracks how often a memory is used (higher = stronger)

# === FUNCTIONS ===

def calculate_histogram_novelty(frame, memory_frames):
    """Compare the histogram of the current frame with stored memories."""
    hist_frame = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist_frame = cv2.normalize(hist_frame, hist_frame).flatten()

    novelty_scores = []
    for mem in memory_frames:
        hist_mem = cv2.calcHist([mem.astype(np.uint8)], [0], None, [256], [0, 256])
        hist_mem = cv2.normalize(hist_mem, hist_mem).flatten()
        novelty_scores.append(cv2.norm(hist_frame, hist_mem, cv2.NORM_L1))

    return np.mean(novelty_scores) if novelty_scores else 0

def calculate_curiosity_reward(frame, memory_frames):
    """Give a reward based on how novel the frame is."""
    novelty = calculate_histogram_novelty(frame, memory_frames)
    reward = novelty * CURIOSITY_FACTOR if novelty > NOVELTY_THRESHOLD else 0
    return reward, novelty

def adaptive_decay(learning_space, recall_mask):
    """Decay learning space slower in recalled areas and faster elsewhere."""
    decay_map = np.ones_like(learning_space) * DECAY_BASE
    decay_map[recall_mask > 0] = 1.0  # Prevent decay in recalled areas
    return learning_space * decay_map

def update_learning_space(learning_space, frame, recall_mask):
    """Blend new frame into the learning space with an adaptive update."""
    frame_float = frame.astype(np.float32)

    # Adaptive decay
    decayed = adaptive_decay(learning_space, recall_mask)

    # Blend new frame, prioritizing novelty
    updated = decayed * (1 - RECALL_BLEND) + frame_float * RECALL_BLEND

    return cv2.GaussianBlur(updated, (3, 3), 0)

def update_long_term_memory(memory, strengths, frame):
    """Update memory: Reinforce useful ones, forget unused ones."""
    frame_float = frame.astype(np.float32)

    if len(memory) < MAX_MEMORY:
        memory.append(frame_float)
        strengths.append(1.0)  # New memories start with strength 1
    else:
        # Compute novelty of new frame
        novelty_new = calculate_histogram_novelty(frame, memory)

        # Compute weakest memory (least recalled)
        min_index = np.argmin(strengths)
        min_strength = strengths[min_index]

        # Replace least-used memory if new one is more novel
        if novelty_new > NOVELTY_THRESHOLD and novelty_new > min_strength:
            memory[min_index] = frame_float
            strengths[min_index] = novelty_new
        else:
            # Reduce strength of unused memories (mimics human forgetting)
            strengths = [s * FORGETTING_RATE for s in strengths]

    return memory, strengths

def recall_memory(memory, strengths, learning_space):
    """Reconstruct memory from past, favoring frequently recalled ones."""
    if not memory:
        return np.zeros_like(learning_space), np.zeros_like(learning_space)

    # Weighted sum of memories, higher strength = more influence
    recall_weighted = sum(m * s for m, s in zip(memory, strengths)) / sum(strengths)
    recall_mask = (recall_weighted > 0.1).astype(np.float32)  # Areas where memory is recalled

    return recall_weighted, recall_mask

# === MAIN LOOP ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1

    # Convert to grayscale and resize
    frame_resized = cv2.resize(frame, LEARNING_SPACE_SIZE, interpolation=cv2.INTER_AREA)
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Recall memory
    recalled_memory, recall_mask = recall_memory(long_term_memory, memory_strengths, learning_space)

    # Update learning space
    learning_space = update_learning_space(learning_space, gray_frame, recall_mask)

    # Calculate curiosity reward
    curiosity_reward, raw_novelty = calculate_curiosity_reward(gray_frame, long_term_memory)
    if curiosity_reward > 0:
        print(f"[Frame {frame_count}] Novelty reward: {curiosity_reward:.3f} (raw novelty: {raw_novelty:.3f})")

    # Update memory with current frame
    long_term_memory, memory_strengths = update_long_term_memory(long_term_memory, memory_strengths, gray_frame)

    # Display
    cv2.imshow('Learning Space', learning_space.astype(np.uint8))
    cv2.imshow('Recalled Memory', recalled_memory.astype(np.uint8))
    cv2.imshow('Current Frame', gray_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

