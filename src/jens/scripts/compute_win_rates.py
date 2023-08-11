import subprocess

with open('main.py', 'r') as f:
    ori_content = f.read()

#for i in range(2_000, 65_000, 5_000):
l = [2_500,5_000,10_000,15_000,20_000,25_000,30_000,35_000]
for i in l:
    edited_content = ori_content.replace('NUMBER_AGENT', str(i))
    with open('main_edited.py', 'w') as f:
        f.write(edited_content)

    subprocess.run(['python3', 'main_edited.py', '--player-2', 'Strong', '--player-1', 'Jens', '--disable-rendering'])

    print(i)
