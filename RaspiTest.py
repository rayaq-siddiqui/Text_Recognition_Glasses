# client code (raspberry pi)
import dropbox 
from picamera import PiCamera
from time import sleep
import requests

# Weather
api_address='http://api.openweathermap.org/data/2.5/weather?appid=0c42f7f6b53b244c78a418f4f181282a&q='
city = 'Waterloo'
url = api_address + city
json_data = requests.get(url).json()
formatted_data = json_data['weather'][0]['description']
current_temp = round(json_data['main']['temp']-273.15)

print(url)
print(formatted_data)
print(str(current_temp)+'°C' )

camera = PiCamera()
camera.start_preview()
# make preview slightly see-through (to see errors)
camera.start_preview(alpha=200)
# sleep for at least 2 seconds before capturing image
sleep(5) 
# Taking pictures
camera.capture('/home/pi/ProjectImages/image.jpg')
camera.stop_preview()
# rotate preview:
camera.rotation = 270

camera.start_preview()
camera.capture('/home/pi/ProjectImages/image.jpg')
camera.stop_preview()

# might have to regenerate access token
dropbox_access_token= "HyKnLMnTXVMAAAAAAAAAARiPgdCGYaDC8-ne9zsm3VbXxXlLSkbihvSjsmiFAqlW"
dropbox_path= "/image.jpg"
computer_path="/home/pi/ProjectImages/image.jpg"
client = dropbox.Dropbox(dropbox_access_token)
print("[SUCCESS] dropbox account linked")

# Upload image to dropbox folder
client.files_upload(open(computer_path, "rb").read(), dropbox_path)
print("[UPLOADED] {}".format(computer_path))


'''
metadata, f = client.files_download(dropbox_path)
out = open("downloadedimage1.jpg", 'wb')
out.write(f.content)
out.close()
'''

# Clear dropbox folder contents
client.files_delete('/image.jpg')
print("[CLEAR] dropbox files deleted")
