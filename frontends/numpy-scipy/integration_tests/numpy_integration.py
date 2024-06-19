
import glob

import subprocess
import multiprocessing
import os
import sys

def run_test_case(test_file):
    print("Running", test_file, end=" ")
    p = subprocess.run( 'cd '+'/'.join(test_file.split('/')[:-1])+' && python3 '+test_file.split('/')[-1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if(p.returncode != 0):
        ret = "FAILED"
        # print(ret)
        # print("=======================================================")
        # print(test_file)
        # failed_tests = failed_tests + 1
        # list_failed_tests.append(test_file)
    else:
        ret = "PASSED"
    print(ret)

    return (test_file, ret, p.stderr.decode())


if __name__ == '__main__':
    categories = ['ops', 'opts', 'kernels', 'compound_exps', 'semiring']
    files = []

    for c in categories:
        files = files + glob.glob("./"+c+"/test_*.py")


    print("\nFound" , len(files), "test cases")

    print("Running the tests with up to {} cores......".format(os.cpu_count()))

    failed_tests = 0
    list_failed_tests = []

    with multiprocessing.Pool() as p:
        results = p.map(run_test_case, files)
        for res in results:
            if(res[1] == "FAILED"):
                list_failed_tests.append((res[0],res[2]))

    print("Passed = ", len(files) - len(list_failed_tests))

    print("Failed = " , len(list_failed_tests))

    if(list_failed_tests):
        print("The following tests failed:")

        for failed_test in list_failed_tests:
            print('  ' , failed_test[0])
    
    if len(sys.argv) == 2:
        if sys.argv[1] == '-v':
            print()
            print("Error messages of failed tests:")
            for failed_test in list_failed_tests:
                print('  ' , failed_test[0])
                print('='*40)
                print(failed_test[1])
                print('*'*40)
    
    if len(list_failed_tests) > 0:
        exit(127)
