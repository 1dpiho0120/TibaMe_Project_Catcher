# 載入爬蟲所需的套件
import requests
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver import Chrome
import time


# 設定瀏覽器與url
driver = Chrome('./爬蟲_HOHO/推薦系統爬蟲/chromedriver')  # 指定瀏覽器的位置
url = "https://listenmood.com/en/i-m-{emotion}-play-me-{style}--{e_num}{s_num}"


# 將網址的變數儲存於list裡面
emotion = ["happy", "in-love", "relax", "romantic"]
e_num = [14, 15, 16, 17]
style = ['acoustic', 'country', 'pop']
s_num = [12, 15, 18]


# 建立一個空list，最後用來放各首歌的list
ytbsong_list = []


# 在每個emotion做迴圈
for x, y in zip(emotion, e_num):

    # 在每種曲風做迴圈
    for i, j in zip(style, s_num):
        # 開啟音樂頁面
        driver.get(url.format(emotion=x, e_num=y, style=i, s_num=j))
        time.sleep(5)

        for i in range(10):
            time.sleep(5)
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # 取得歌名
            song_list = soup.select('span[id="trackName"]')
            for song_soup in song_list:
                song = song_soup.text

            # 取得歌曲網址
            ytb_list = soup.select('a[id="open-on-youtube"]')
            for ytb_url_soup in ytb_list:
                ytb_url = ytb_url_soup['href']

            s_list = [x, song, ytb_url]

            if song not in ytbsong_list:
                ytbsong_list.append(s_list)
            else:
                break

            driver.find_element_by_id('next-track').click()


# print(ytbsong_list)


df = pd.DataFrame(ytbsong_list, columns=["Mood", "Title", "Content"])
print(df)

# 將DataFrame寫入csv
df.to_csv('./推薦系統資料.csv', header=True)
