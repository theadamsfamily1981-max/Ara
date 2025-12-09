# Makefile for A10PED Neuromorphic AI Tile Driver
# Linux kernel module

obj-m += a10ped_driver.o

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a

load:
	sudo insmod a10ped_driver.ko

unload:
	sudo rmmod a10ped_driver

reload: unload load

.PHONY: all clean install load unload reload
