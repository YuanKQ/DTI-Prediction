# Source Code for Graduation Project

同步命令：

rsync -arvz DDISuccess cdy@219.223.195.174:/home/cdy/ykq/  --exclude-from='DDISuccess/exclude.list'

rsync -arvz DDISuccess kqyuan@219.223.197.143:/home/kqyuan/  --exclude-from='DDISuccess/exclude.list'

rsync -arvz model/*.py lei@219.223.194.155:/home/lei/DDISuccess/model  --exclude-from='exclude.list'
rsync -arvz Data/DTI/*.pickle lei@219.223.194.155:/home/lei/DDISuccess/Data/DTI  --exclude-from='exclude.list'
rsync -arvz util/*.py lei@219.223.194.155:/home/lei/DDISuccess/util  --exclude-from='exclude.list'



