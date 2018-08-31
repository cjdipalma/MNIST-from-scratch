i = 0

def inc():
    global i
    i += 1

def pr():
    global i
    print(i)
    i += 1
    #i = 1

if __name__ == '__main__':
    i = 1
    inc()
    pr()
    pr()
