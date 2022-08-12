speeds = []
while True:
    try:
        line = input()
        if line == '':
            if speeds:
                print("Average Speed: {:.2f}".format(sum(speeds) / len(speeds)))
                speeds = []
            continue
        start_index = line.find("Speed:")
        end_index = line.find("samples/sec")
        line = line[start_index + 7: end_index - 1]
        speeds.append(float(line))
    except:
        break
