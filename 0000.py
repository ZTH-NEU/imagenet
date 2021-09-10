import os
import shutil
a = os.walk('E:\\data\\test_1')
file_name_1 = 'E:\\data\\input\\val\\benign'
file_name_2 = 'E:\\data\\input\\val\\malignant'

print(a)
list_file1 = []
list_file2 = []
count_a = 0
count_b = 0
count_c = 0
for maindir, subdir, file_name_list in a:
    print(len(file_name_list))
    for f in file_name_list:
        # print(f.split('-')[-1].split('.')[0] )
        file_name = os.path.join(maindir, f)
        if int(f.split('-')[-1].split('.')[0]) == 0:
            shutil.copy(file_name, file_name_1)
            # os.system('copy %s %s' % (file_name, file_name_1))
            count_a += 1
        elif int(f.split('-')[-1].split('.')[0]) == 1:
            shutil.copy(file_name, file_name_2)

            # os.system('copy %s %s' % (file_name, file_name_2))
            count_b += 1
        else:
            count_c += 1
            print(file_name)
    print(count_a,count_b)
    print(count_c)

# 1.数据泄露
# 2.resnet-50 BN 实验
# 3.数据处理
# 4.roc曲线重新绘制



