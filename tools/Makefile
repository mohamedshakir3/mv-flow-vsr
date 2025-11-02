APP = extract_mvs
SRC = extract_mvs.c

PKGS = libavformat libavcodec libavutil
CFLAGS  = $(shell pkg-config --cflags $(PKGS))
LDLIBS  = $(shell pkg-config --libs   $(PKGS))

$(APP): $(SRC)
	$(CC) $(SRC) -o $(APP) $(CFLAGS) $(LDLIBS)

clean:
	rm -f $(APP)