import vlc
player=vlc.MediaPlayer('rtsp://admin:jg00dman@192.168.1.114:554/cam/realmonitor?channel=1&subtype=0')
player.play()