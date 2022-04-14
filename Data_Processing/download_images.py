import paramiko
import os
from stat import S_ISDIR as isdir

#define your local paramter
host_name='100.26.215.145'
user_name='ubuntu'
key_filename='/Users/dool/.ssh/VPC-MOJAHID-PRI.pem'
remote_dir = '/home/ubuntu/Notebooks'
local_dir = '/Users/dool/project/data'

def down_from_remote(sftp_obj, remote_dir_name, local_dir_name):
    "" "download files remotely" ""
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
