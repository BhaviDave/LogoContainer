import requests
url = 'http://127.0.0.1:5000/im_labels'
my_img = {'image': open(r'C:\Users\bhavidave\Desktop\Axle\Logo\DeepLogo\flickr_logos_27_dataset\flickr_logos_27_dataset_images\flickr_logos_27_dataset_images\2962045.jpg', 'rb')}
r = requests.post(url, files=my_img)

# convert server response into JSON format.
print(r.json())
#ref
#https://jdhao.github.io/2020/04/12/build_webapi_with_flask_s2/