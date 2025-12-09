# Tools Makefile

CC = gcc
CFLAGS = -Wall -Werror -O2 -I../include
LDFLAGS = -L../api -lsnn -lpthread

TOOLS = monitoring/snn_monitor

all: $(TOOLS)

monitoring/snn_monitor: monitoring/snn_monitor.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TOOLS)

.PHONY: all clean
