# Compiler and Tools
CC       := gcc
TARGET   := cFVC
SRC      := fragmentVectorComparison.c

# Compiler Flags
CSTD     := -std=gnu2x
CFLAGS   := -Wall -Wextra -pedantic -Wmissing-prototypes -Wstrict-prototypes \
            -Wold-style-definition -Wconversion -Wshadow -Wundef -Wpointer-arith \
            -Wfloat-equal -Wbad-function-cast -Wformat=2 -fstack-protector-strong \
            -D_GNU_SOURCE -D_FORTIFY_SOURCE=2 -Werror -O3 -march=native -flto -fbounds-check $(CSTD)

# Linker Flags
LDFLAGS  := -lm -fopenmp

# Build Rules
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Clean Rule
clean:
	rm -f $(TARGET)

.PHONY: all clean

