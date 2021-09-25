import os
import subprocess


path = os.getcwd()
MOT16path = []
os.environ['path'] = 'C:\\Users\\Kenny\\anaconda3\\envs\\pytorch\\lib\\site-packages\\ffmpeg\\_run.py'

path += '\\data\\MOT16\\train'

def find_dir(dir):
    fds = os.listdir(dir)
    for fd in fds:
        full_path=os.path.join(dir,fd)

        # if your system is linux you need to comment out this code
        # windows / linux \
        # full_path = full_path.replace('\\', '/')

        if os.path.isdir(full_path):
            # print("資料夾: ", full_path)
            MOT16path.append(full_path)
            # find_dir(full_path)
        else:
            print('檔案: ', full_path)
find_dir(path)

# print(MOT16path)


for p in MOT16path:

    Input = p + '\\img1\\%6d.jpg'
    result = p.split('\\')
    name = result[-1]
    name += '.mp4'
    # print(name)

    del result[-6:]
    Output = '\\'.join(result)
    Output += '\\inference\\input\\' + name
    # print(Input)
    # print(Output)

    # print('ffmpeg -f image2 -i {0} {1}'.format(Input, Output))


    # os.system('ffmpeg -f image2 -i {0} {1}'.format(Input, Output))
Out = 'C:\\Users\\Kenny\\PycharmProjects\\Yolov5_DeepSort_Pytorch\\inference\\output'

source = 'C:\\Users\\Kenny\\PycharmProjects\\Yolov5_DeepSort_Pytorch\\inference\\input'

if os.path.isdir(source):
    fds = os.listdir(source)
    txt_file_name = fds[1].split('\\')[-1].split('.')[0]
    txt_path = str(Out) + '\\' + txt_file_name + '.txt'
else:

    txt_file_name = source.split('\\')[-1].split('.')[0]
    txt_path = str(Out) + '\\' + txt_file_name + '.txt'
print(txt_file_name)
print(txt_path)