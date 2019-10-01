import os
import sys
import pathlib
import re

def replace(dirpath, old_words, new_words):
    if not dirpath or not old_words or not new_words:
        raise Exception('Argument is null or empty')
    
    p = pathlib.Path(dirpath)
    htmlfiles = list(p.glob('*.htm'))
    pattern = re.compile(old_words)
    for f in p.glob('*.htm'):
        with f.open('r+', encoding='gb2312') as handler:
            content = handler.read()
            if pattern.search(content):
                content = pattern.sub(new_words, content)
                handler.seek(0)
                handler.write(content)
                handler.truncate()


if __name__ == '__main__':
    # if len(sys.argv) != 4:
    #     print('Input path, old words and new words...')
    # else:
    #     print('path{0} | oldwords{1} | newwords{2}').format(sys.argv[1], sys.argv[2], sys.argv[3])
    #     replace(sys.argv[1], sys.argv[2], sys.argv[3])
    abspath = os.path.dirname(__file__)
    docpath = os.path.join(abspath, 'ys-yashen/')
    oldwords = '沪ICP备05019063号-3'
    newwords = '浙ICP备19034128号-1'
    replace(docpath, oldwords, newwords)

    oldwords = 'http://www.miitbeian.gov.cn'
    newwords = 'http://www.beian.miit.gov.cn'
    replace(docpath, oldwords, newwords)

    
