import imageio

for i in range(10):
    gif = []
    path = f"samples/{i}"

    for filename in range(100):
            gif.append(imageio.imread(f"{path}/sample_epoch_{filename}.jpg"))
    
    imageio.mimsave(f"gifs/{i}.gif", gif, fps=24)