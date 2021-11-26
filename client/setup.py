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
os.chdir('federated_aia_test')
print(os.listdir())

# learning.cfg
class client_env_setup():
    def __init__(self):
        self.key = self.generate_key()
        self.save_cfg()
    def generate_key(self):
        import secrets
        key = secrets.token_urlsafe(16)
        return key
    
    def save_cfg(self):
        print('Create local.cfg')
        with open('./local.cfg', mode='w+', encoding='UTF-8') as f:
            r = f.readline()
#             print(r)
            f.writelines('key='+self.key)
# generate_key()
if 'local.cfg' in os.listdir():
    print('local.cfg already existed')
    pass
else:
    a = client_env_setup()


# get API-KEY from learning.cfg
with open('./local.cfg', mode='r', encoding='UTF-8') as f:
    r = f.readlines()
    item_dict = {}
    for item in r:
        key, value = item.split('=')
        item_dict[key] = value
    print(r)
    
API_KEY = item_dict['key']



cmd_lis = '''git branch {key}
git checkout {new_branch}
rm .gitignore
mkdir ../tmp
mv * ../tmp
git add .
git commit -m'new branch for client'
git push https://at102091:12345678@gitlab.aiacademy.tw/junew/federated_aia_test.git
'''.format(key=API_KEY, new_branch=API_KEY)
run_cmd(cmd_lis)

if '.gitignore' not in os.listdir():
    with open('.gitignore', mode='w+', encoding='UTF-8') as f:
        f.writelines('model-logs\n')
        f.writelines('__pycache__\n')
        f.writelines('.ipynb_checkpoints\n')
        for i in os.listdir('../tmp'):
            print(i)
            f.writelines(i+'\n')

            
            
cmd_lis = '''mv ../tmp/* .
rm -rf ../tmp
git add .
git commit -m'new branch for client'
git push https://at102091:12345678@gitlab.aiacademy.tw/junew/federated_aia_test.git
'''   
run_cmd(cmd_lis)
# print(os.listdir())
# run_cmd(cmd_lis)