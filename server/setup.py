import os

run_cmd = lambda cmd_lis:[os.popen(i).read() for i in cmd_lis.split('\n')]

if 'federated_aia_test' in os.listdir():
    cmd_lis = '''cd federated_aia_test
    git pull
    '''
else: 
    print('update federated_aia_test')
    cmd_lis = '''git clone https://at102091:12345678@gitlab.aiacademy.tw/junew/federated_aia_test.git
    cd federated_aia_test
    git pull
    '''

run_cmd(cmd_lis)