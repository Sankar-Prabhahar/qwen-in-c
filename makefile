CC=gcc
CFLAGS=-O3 -std=c99 -Wall -Wextra -mavx2 -mfma -fopenmp -Iinclude
LDFLAGS=-fopenmp

SRC=$(wildcard src/*.c)
OBJ=$(SRC:.c=.o)

TARGET=qwen30b

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f src/*.o $(TARGET)