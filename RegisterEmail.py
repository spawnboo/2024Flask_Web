# coding=utf-8
from email.mime.text import MIMEText
import smtplib
import random

# 生成六碼驗證碼
def generate_verify_code():
    code = ""
    for _ in range(6):
        code += str(random.randint(0,9))
    return code

# 會員註冊驗證系統
def Register_Function(register_email):
    try:
        # 生成驗證碼
        verify_code = generate_verify_code()

        # 連結郵件伺服器,登入
        server=smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login("e01646166@gmail.com", "skqtvprviyrqmmtk")
        # 寄件者與收件者
        from_addr = 'e01646166@gmail.com'
        to_addr =   register_email

        # 寄件內容
        msg = '這是鱷魚動物園的驗證系統,您的驗證碼是:\n \n'
        msg += str(verify_code) + "\n \n"
        msg += "此驗證碼只在當前頁面生效,離開後請重新申請! \n"
        msg += "此信件為自動寄出, 請勿回信給該信箱! \n"
        mime = MIMEText(msg, "plain", "utf-8")

        # 顯示名稱
        mime["Subject"] = "鱷魚動物園 驗證碼寄送"
        mime["From"] = "鱷魚管家 小布"
        mime["To"] = to_addr
        # 寄出郵件 + 關閉方法
        server.sendmail(from_addr, to_addr, mime.as_string())
        server.close()
        return verify_code

    except Exception as e:
        print(str(e))
        return str(e)


if __name__ == '__main__':
    # verify_code = generate_verify_code()
    # print(verify_code)
    register_email = 'e01646166@hotmail.com'
    Register_Function(register_email)
