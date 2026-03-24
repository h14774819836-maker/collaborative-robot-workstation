import json
import math
import threading
import time
from socket import *
# endregion
class Blinx_Jaka_Rev_Socket():
    def __init__(self, public_class):
        self.max_attempts = 2 # 设置最大尝试次数
        self.attempt_count = 0  # 初始化尝试次数计数器
        self.public_class = public_class
        self.jaka_connect_sucess = False
        while self.attempt_count < self.max_attempts:
            try:
                self.socket_jaka = socket(AF_INET, SOCK_STREAM)
                self.socket_jaka.setsockopt(SOL_SOCKET, SO_KEEPALIVE, 1)  # 在客户端开启心跳维护
                self.socket_jaka.connect((str(self.public_class.jaka_ip), int(self.public_class.jaka_port_10000)))
                self.BUF_SIZE = 10000
                # 开线程，防止堵塞
                self.jaka_t = threading.Thread(target=self.blinx_jaka_rev_msg)
                self.jaka_t.start()
                break
            except Exception as e:
                self.socket_jaka.close()
                self.attempt_count += 1  # 增加尝试次数
                if self.attempt_count >= self.max_attempts:
                    self.jaka_connect_sucess = False
                    print("机器人连接失败超过最大尝试次数，停止尝试")
                    break

    # 通讯关闭方法
    def blinx_jaka_close(self):
        self.socket_jaka.close()

    # 消息接收
    def blinx_jaka_rev_msg(self):
        while True:
            time.sleep(0.01)
            try:
                # 消息接受
                data = self.socket_jaka.recv(self.BUF_SIZE)
                if len(data) > 0:
                    jason_data = json.loads(data.decode())
                    if "eip_adpt_ain" in str(data):
                        self.public_class.ai_value = jason_data['eip_adpt_ain']
            except Exception as e:
                print("blinx_jaka_rev_msg: ", e)
                continue
