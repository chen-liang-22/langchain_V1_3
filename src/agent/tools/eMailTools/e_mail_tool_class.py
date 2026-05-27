import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

from langchain_core.tools import BaseTool
from pydantic import create_model, Field


class MyWebSearchTool(BaseTool):
    name: str = "发送邮件"
    description: str = "发送邮件到指定收件人"
    args_schema: type = create_model(
        "SendEmailInput",
        to=(str, Field(..., description="收件人的电子邮箱地址")),
        subject=(str, Field(..., description="邮件的主题（标题）")),
        body=(str, Field(..., description="邮件正文内容，纯文本形式")),
    )

    def _run(self, to: str, subject: str, body: str) :
        # 构造邮件对象
        msg = EmailMessage()
        msg["From"] = os.getenv("SMTP_USER", "no-reply@example.com")
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)

        # 读取SMTP配置（推荐使用环境变量）
        smtp_server = os.getenv("SMTP_HOST", "smtp.example.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

        try:
            if use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls(context=context)
                    if smtp_user and smtp_pass:
                        server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    if smtp_user and smtp_pass:
                        server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
        except smtplib.SMTPException as e:
            logging.error("邮件发送失败，收件人 %s，发件人 %s", to, msg["From"], exc_info=True)
            logging.error("邮件发送失败，打印发件方信息 → %s", to, exc_info=True)
            raise RuntimeError(f"邮件发送失败: {e}") from e

        return f"邮件已发送至 {to}"