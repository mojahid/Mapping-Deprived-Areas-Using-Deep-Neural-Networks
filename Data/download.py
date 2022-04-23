import paramiko
import os
from stat import S_ISDIR as isdir

# get your local directory
city_name='lagos'#accra
local_dir = os.getcwd()
#will cretae folder with city name and download your data
local_dir = os.path.join(local_dir,city_name)
#print(local_dir)

#define your parameters
host_name='44.202.128.187'
user_name='ubuntu'
key_filename='/Users/mojahid/.ssh/World_Bank.pem'

#this will not be changed
remote_dir = '/home/ubuntu/Autoencoder/Autoencoder/Accra_png/Train_png/0'

def down_from_remote(sftp_obj, remote_dir_name, local_dir_name):
    #download files remotely
    remote_file = sftp_obj.stat(remote_dir_name)
    if isdir(remote_file.st_mode):
        #Folder, can't download directly, need to continue cycling
        check_local_dir(local_dir_name)
        print('Start downloading folder: '+ remote_dir)
        for remote_file_name in sftp.listdir(remote_dir_name):
            sub_remote = os.path.join(remote_dir_name, remote_file_name)
            sub_remote = sub_remote.replace('\\', '/')
            sub_local = os.path.join(local_dir_name, remote_file_name)
            sub_local = sub_local.replace('\\', '/')
            down_from_remote(sftp_obj, sub_remote, sub_local)
    else:
        #Files, downloading directly
        print('Start downloading file: '+ remote_dir_name)
        sftp.get(remote_dir_name, local_dir_name)

#check if the folder is not exists then cretaed
def check_local_dir(local_dir_name):

    if not os.path.exists(local_dir_name):
       os.makedirs(local_dir_name)


if __name__ == "__main__":
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host_name, username=user_name, key_filename=key_filename)
    sftp = ssh.open_sftp()

    #Remote file start download
    down_from_remote(sftp, remote_dir, local_dir)

    #Close connection
    ssh.close()