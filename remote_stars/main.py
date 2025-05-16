import socket
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

server_address = "84.237.21.36"
server_port = 5152

def receive_exact(sock, num_bytes):
    buffer = bytearray()
    while len(buffer) < num_bytes:
        part = sock.recv(num_bytes - len(buffer))
        if not part:
            return None
        buffer.extend(part)
    return buffer

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
    connection.connect((server_address, server_port))
    response = b"nope"
    
    plt.ion()
    plt.figure()
    
    while response != b"yep":
        connection.send(b"get")
        payload = receive_exact(connection, 40002)
        print("Bytes received:", len(payload))

        img_data = np.frombuffer(payload[2:40002], dtype=np.uint8).reshape(payload[0], payload[1])
        
        thresholded = img_data > 100
        labeled = label(thresholded)
        objects = regionprops(labeled)
        
        if len(objects) < 2:
            print("Objects have overlapped.")
            continue
        
        coord_a, coord_b = objects[0].centroid, objects[1].centroid
        distance = ((coord_b[0] - coord_a[0])**2 + (coord_b[1] - coord_a[1])**2) ** 0.5
        
        connection.send(f"{round(distance, 1)}".encode())
        print("Coordinates:", coord_a, coord_b, "Distance:", round(distance, 1))
        print(connection.recv(10))
        
        plt.clf()
        plt.imshow(img_data)
        plt.pause(1)

        connection.send(b"beat")
        response = connection.recv(10)
