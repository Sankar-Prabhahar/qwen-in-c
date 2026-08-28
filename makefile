CC=gcc
CFLAGS=-O3 -std=c99 -Wall -Wextra -mavx2 -mfma -Iinclude

SRC=$(wildcard src/*.c)
OBJ=$(SRC:.c=.o)

TARGET=qwen30b

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f src/*.o $(TARGET)