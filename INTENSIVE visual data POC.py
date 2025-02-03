import cv2
import numpy as np

# === PARAMETERS ===
# Learning space parameters
LEARNING_SPACE_SIZE = (512, 512)  # (width, height)
DECAY_LEARNING = 0.98             # How quickly the learning space fades (per frame)

# Advanced blending parameters
BASE_BLEND = 0.90                 # Base weight for the existing learning space
NOVEL_BLEND = 0.10                # Additional weight given to areas deemed novel
DIFF_THRESHOLD = 30               # Pixel difference threshold (0-255) to consider a change as novel

# Novelty/reward parameters
PREDICTION_THRESHOLD = 0.5        # Threshold for histogram difference to be considered novel
CURIOSITY_FACTOR = 0.1            # Multiplier for the novelty to compute reward

# Long-term memory parameters
DECAY_FACTOR = 0.99               # Factor by which old memories fade
MAX_MEMORY = 10                   # Maximum number of images stored in memory

# === INITIAL SETUP ===
# Create an empty learning space (grayscale image in float32).
learning_space = np.zeros(LEARNING_SPACE_SIZE, dtype=np.float32)
# Long-term memory: each element will be a grayscale image.
long_term_memory = []

# === HELPER FUNCTIONS ===

def calculate_histogram_novelty(current_frame, memory_frames):
    """
    Calculate novelty by comparing the histogram of the current frame
    to those of each stored memory frame. Returns the average L1 norm
    of the histogram differences.
    """
    # current_frame is assumed to be uint8.
    hist_current = cv2.calcHist([current_frame], [0], None, [256], [0, 256])
    hist_current = cv2.normalize(hist_current, hist_current).flatten()

    novelty_scores = []
    for mem_frame in memory_frames:
        # Ensure the memory frame is in uint8 for histogram calculation.
        mem_uint8 = mem_frame.astype(np.uint8)
        hist_mem = cv2.calcHist([mem_uint8], [0], None, [256], [0, 256])
        hist_mem = cv2.normalize(hist_mem, hist_mem).flatten()
        # Use L1 norm (sum of absolute differences) as the difference measure.
        diff = cv2.norm(hist_current, hist_mem, cv2.NORM_L1)
        novelty_scores.append(diff)

    return np.mean(novelty_scores) if novelty_scores else 0

def calculate_curiosity_reward(current_state, memory_frames):
    """
    Calculate a curiosity reward based on histogram novelty.
    Returns both the reward and the raw novelty score.
    """
    novelty = calculate_histogram_novelty(current_state, memory_frames)
    reward = novelty * CURIOSITY_FACTOR if novelty > PREDICTION_THRESHOLD else 0
    return reward, novelty

def advanced_explore_learning_space(learning_space, new_frame):
    """
    Update the learning space with a more advanced method:
      1. Decay the current learning space slightly.
      2. Compute the pixelwise absolute difference between the new frame
         and the current learning space.
      3. Create a mask from differences that exceed a set threshold.
      4. Update the learning space more strongly in regions of high novelty.
      5. Optionally, apply smoothing (Gaussian blur) for a network-like appearance.
    """
    # Convert new frame to float32.
    new_frame_float = new_frame.astype(np.float32)
    
    # Decay the current learning space.
    decayed = learning_space * DECAY_LEARNING
    
    # Compute the pixelwise difference.
    diff = cv2.absdiff(new_frame_float, decayed)
    
    # Create a mask where the difference exceeds DIFF_THRESHOLD.
    # This mask will have values 0 or 1.
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 1, cv2.THRESH_BINARY)
    
    # Calculate a weighted update:
    # In areas of high difference (mask==1), give an extra boost from the new frame.
    updated = decayed * BASE_BLEND + new_frame_float * (1 - BASE_BLEND + mask * NOVEL_BLEND)
    
    # Optionally, smooth the result for a more network-like, cohesive appearance.
    updated = cv2.GaussianBlur(updated, (3, 3), 0)
    
    return updated

def update_long_term_memory(memory, current_state):
    """
    Update the long-term memory with the current state.
    All stored memories are decayed first.
    If there's room (< MAX_MEMORY), the current state is added.
    When at capacity, the new state’s novelty is computed relative to all memories,
    and if it's more novel than the least "important" memory, it replaces that memory.
    """
    # Decay each memory (they are stored as float32).
    decayed_memory = [state * DECAY_FACTOR for state in memory]

    # Convert current_state to float32 for comparisons.
    current_float = current_state.astype(np.float32)

    if len(decayed_memory) < MAX_MEMORY:
        decayed_memory.append(current_float)
        print(f"Memory added. Total memories: {len(decayed_memory)}.")
    else:
        # Compute the average L1 norm between the new state and each stored memory.
        novelty_new = np.mean([
            cv2.norm(current_float, state, cv2.NORM_L1) for state in decayed_memory
        ])

        # Compute an "importance" score for each stored memory:
        importance_list = []
        for i, state in enumerate(decayed_memory):
            differences = [
                cv2.norm(state, other, cv2.NORM_L1)
                for j, other in enumerate(decayed_memory) if i != j
            ]
            importance = np.mean(differences) if differences else 0
            importance_list.append(importance)
        
        # Identify the memory with the lowest importance.
        min_index = np.argmin(importance_list)
        min_importance = importance_list[min_index]

        # Replace the least important memory if the new state is more novel.
        if novelty_new > min_importance:
            decayed_memory[min_index] = current_float
            print(f"Replaced memory at index {min_index}: New novelty {novelty_new:.2f} > old importance {min_importance:.2f}.")
        else:
            print(f"New state novelty ({novelty_new:.2f}) did not exceed minimum stored importance ({min_importance:.2f}). No replacement.")

    return decayed_memory

# === MAIN LOOP ===

# Set up video capture from the default webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit(1)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1

    # Resize frame and convert to grayscale.
    frame_resized = cv2.resize(frame, LEARNING_SPACE_SIZE, interpolation=cv2.INTER_AREA)
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Update the learning space with the advanced exploration function.
    learning_space = advanced_explore_learning_space(learning_space, gray_frame)

    # Calculate curiosity reward based on histogram novelty.
    curiosity_reward, raw_novelty = calculate_curiosity_reward(gray_frame, long_term_memory)
    if curiosity_reward > 0:
        print(f"[Frame {frame_count}] Novelty reward: {curiosity_reward:.3f} (raw novelty: {raw_novelty:.3f})")

    # Update the long-term memory with the current frame.
    long_term_memory = update_long_term_memory(long_term_memory, gray_frame)

    # Display the learning space and the current frame.
    cv2.imshow('Learning Space', learning_space.astype(np.uint8))
    cv2.imshow('Current Frame', gray_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup.
cap.release()
cv2.destroyAllWindows()

