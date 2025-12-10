# AraVoice Audio Kernel - Makefile
# =================================

CC = gcc
CFLAGS = -Wall -Wextra -O2 -I./include
LDFLAGS = -lportaudio -lm -lpthread

SRC_DIR = src
INCLUDE_DIR = include
EXAMPLES_DIR = examples
BUILD_DIR = build

# Library source
LIB_SRC = $(SRC_DIR)/audio_portaudio.c
LIB_OBJ = $(BUILD_DIR)/audio_portaudio.o

# Examples
EXAMPLES = sine_test ringbuf_test

.PHONY: all clean examples install

all: $(BUILD_DIR) $(LIB_OBJ) examples

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_OBJ): $(LIB_SRC) $(INCLUDE_DIR)/audio.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

examples: $(EXAMPLES:%=$(BUILD_DIR)/%)

$(BUILD_DIR)/sine_test: $(EXAMPLES_DIR)/sine_test.c $(LIB_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/ringbuf_test: $(EXAMPLES_DIR)/ringbuf_test.c $(LIB_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

# Install PortAudio on Debian/Ubuntu
install-deps:
	sudo apt-get update
	sudo apt-get install -y portaudio19-dev

# Quick test
test: $(BUILD_DIR)/sine_test
	./$(BUILD_DIR)/sine_test

# Help
help:
	@echo "AraVoice Audio Kernel"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build library and examples"
	@echo "  examples     - Build example programs"
	@echo "  clean        - Remove build artifacts"
	@echo "  install-deps - Install PortAudio (Debian/Ubuntu)"
	@echo "  test         - Run sine wave test"
	@echo ""
	@echo "Examples:"
	@echo "  ./build/sine_test       - Play 440Hz tone"
	@echo "  ./build/sine_test 880   - Play 880Hz tone"
	@echo "  ./build/ringbuf_test    - Test async ring buffer"
