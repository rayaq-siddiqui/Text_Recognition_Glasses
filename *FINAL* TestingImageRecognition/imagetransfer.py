# client code (raspberry pi)
import dropbox

f = open('dropbox/output.txt', "rb")
dbx = dropbox.Dropbox('HyKnLMnTXVMAAAAAAAAAARiPgdCGYaDC8-ne9zsm3VbXxXlLSkbihvSjsmiFAqlW')
dbx.files_upload(f.read(), 'output/out.txt')
f.close()

# ***********
# TESTING
# dropbox_access_token = "HyKnLMnTXVMAAAAAAAAAARiPgdCGYaDC8-ne9zsm3VbXxXlLSkbihvSjsmiFAqlW"
# client = dropbox.Dropbox(dropbox_access_token)
# computer_path = "img/dropbox/input.jpeg"
#
# # where it will be saved on dropbox
# dropbox_path = "/input/in.jpeg"
#
# # this will save the code on my computer
# _, f = client.files_download(dropbox_path)
# out = open("dropbox/input.jpeg", 'wb')
# out.write(f.content)
# out.close()
# **********************

# # might have to regenerate access token
# # access token

# delete the previous file
# client.files_delete(dropbox_path)

# client.files_upload(open(computer_path, "rb").read(), dropbox_path)

# # MIGHT USE LATER
#
# # _, g = client.files_download(dropbox_path)
# # out = open("dropbox/input.jpeg", 'wb')
# # out.write(g.content)
# # out.close()
#
# # ACTUAL STUFF HERE
# dropbox_access_token = "HyKnLMnTXVMAAAAAAAAAARiPgdCGYaDC8-ne9zsm3VbXxXlLSkbihvSjsmiFAqlW"
# dbx = dropbox.Dropbox(dropbox_access_token)
# print("[SUCCESS] dropbox account linked")
#
# # change computer path as required
# computer_path = "img/input/arabic2.jpeg"
# dropbox_path = "/input/in.jpeg"
#
# f = open(computer_path)
#
# file_to = '/output/arabic.jpeg'
# # dbx.files_delete(file_to)
# dbx.files_upload(f, file_to)
# print("[UPLOADED] {}".format(computer_path))
#
# f.close()    # close it from before


