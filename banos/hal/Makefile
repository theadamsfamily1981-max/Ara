# Ara Somatic HAL - Build
#
# Builds libara_somatic.so for C/C++ integration

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -I.
LDFLAGS = -shared -lrt

TARGET = libara_somatic.so
STATIC = libara_somatic.a
SOURCES = ara_somatic.c
OBJECTS = $(SOURCES:.c=.o)
HEADERS = ara_somatic.h

.PHONY: all clean install test

all: $(TARGET) $(STATIC)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(STATIC): $(OBJECTS)
	ar rcs $@ $^

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET) $(STATIC)

install: $(TARGET) $(STATIC)
	install -d $(DESTDIR)/usr/local/lib
	install -d $(DESTDIR)/usr/local/include
	install -m 644 $(TARGET) $(DESTDIR)/usr/local/lib/
	install -m 644 $(STATIC) $(DESTDIR)/usr/local/lib/
	install -m 644 $(HEADERS) $(DESTDIR)/usr/local/include/
	ldconfig || true

# Simple test program
test: $(TARGET)
	@echo "Building test..."
	$(CC) -o test_somatic test_somatic.c -L. -lara_somatic -lrt
	@echo "Run: LD_LIBRARY_PATH=. ./test_somatic"

# Test program source
test_somatic.c:
	@echo '#include <stdio.h>' > $@
	@echo '#include "ara_somatic.h"' >> $@
	@echo 'int main() {' >> $@
	@echo '    ara_somatic_t *som = ara_somatic_open();' >> $@
	@echo '    if (!som) { perror("open"); return 1; }' >> $@
	@echo '    printf("Somatic magic: 0x%08X\\n", som->magic);' >> $@
	@echo '    printf("Version: %u\\n", som->version);' >> $@
	@echo '    printf("State: %s\\n", ara_state_name(som->system_state));' >> $@
	@echo '    printf("Pain: %u\\n", som->fpga.pain_level);' >> $@
	@echo '    ara_somatic_close(som);' >> $@
	@echo '    return 0;' >> $@
	@echo '}' >> $@
