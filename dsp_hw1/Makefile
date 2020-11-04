.PHONY: all clean run
CC=g++
CFLAGS= -std=c++11 -g
LDFLAGS=-lm
TARGET=train test try
TRAIN_ITER=100

all: $(TARGET)

train: src/train.cpp
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) -Iinc

try: src/try.cpp
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) -Iinc

test: src/test.cpp
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) -Iinc

models.o: inc/models.h src/models.cpp
	$(CC) -c $@ $^ $(CFLAGS) $(LDFLAGS) -Iinc

clean:
	rm -f $(TARGET)

