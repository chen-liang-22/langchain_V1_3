from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from agent.my_llm import *
import random
import json

# 随机国家池，覆盖主要外贸市场
COUNTRY_POOL = [
    "埃及", "沙特阿拉伯", "巴西", "墨西哥", "印度尼西亚",
    "尼日利亚", "土耳其", "越南", "泰国", "阿根廷",
    "肯尼亚", "巴基斯坦", "孟加拉国", "伊朗", "哈萨克斯坦",
    "乌兹别克斯坦", "摩洛哥", "坦桑尼亚", "秘鲁", "菲律宾"
]

# 客户画像生成

customer_profile = {
    "industry_label": "农业机械",
    "core_focus_label": "行业趋势",
    "personality_label": "直接果断",
    "purchase_stage_label": "谈判中",
    "secondary_focus_label": "技术创新",
    "cooperation_stage_label": "老客户复购",
    "customer_category_label": "B端",
    "price_sensitivity_label": "极高",
    "applicable_product_label": "外综服"
}

prompt_template = PromptTemplate.from_template(
    """你是一个外贸销售培训专家，请根据以下客户标签，生成一个真实感强的客户画像实例，用于销售对话训练。

严格要求：
- 300字以内
- 输出为一段连续文字，不得有任何换行
- 客户必须来自：{country}，使用符合该国文化的真实姓名
- 必须包含：所在城市、公司规模与主营业务
- 必须包含：当前采购背景与诉求（结合采购阶段和合作阶段）
- 必须包含：沟通风格描述（结合性格特征）
- 必须包含：一句典型话术，自然融入正文，不加引号、不加标点强调
- 禁止出现任何提示性语句，如"可额外补充"、"自定义信息"等
- 直接输出人物描述，不加标题、编号、分隔线

客户标签：
- 行业：{industry_label}
- 客户分类：{customer_category_label}
- 合作阶段：{cooperation_stage_label}
- 采购阶段：{purchase_stage_label}
- 核心关注点：{core_focus_label}
- 次要关注点：{secondary_focus_label}
- 价格敏感度：{price_sensitivity_label}
- 适用产品：{applicable_product_label}
- 性格特征：{personality_label}"""
)

# 校验 prompt：让模型对照标签逐项判断实例是否合格
validate_template = PromptTemplate.from_template(
    """你是一个外贸销售培训质检专家，请对以下客户画像实例进行校验。

客户标签要求：
- 行业：{industry_label}
- 客户分类：{customer_category_label}
- 合作阶段：{cooperation_stage_label}
- 采购阶段：{purchase_stage_label}
- 核心关注点：{core_focus_label}
- 次要关注点：{secondary_focus_label}
- 价格敏感度：{price_sensitivity_label}
- 适用产品：{applicable_product_label}
- 性格特征：{personality_label}

待校验的客户画像实例：
{instance}

请逐项校验实例是否符合标签要求，输出严格的JSON格式，不要有任何多余文字：
{{
  "passed": true或false,
  "score": 0到100的整数,
  "checks": {{
    "has_name": true或false,
    "has_city": true或false,
    "has_company_info": true或false,
    "has_purchase_background": true或false,
    "has_communication_style": true或false,
    "has_typical_phrase": true或false,
    "matches_industry": true或false,
    "matches_cooperation_stage": true或false,
    "matches_purchase_stage": true或false,
    "matches_personality": true或false,
    "no_template_leakage": true或false
  }},
  "issues": ["问题描述1", "问题描述2"]
}}"""
)

chain = prompt_template | llm | StrOutputParser()
validate_chain = validate_template | llm | StrOutputParser()


def generate_profile(profile: dict) -> str:
    country = random.choice(COUNTRY_POOL)
    result = chain.invoke({**profile, "country": country})
    result = result.replace("\n", "").replace('"', '').replace('"', '').replace('"', '').strip()
    return result


def validate_profile(instance: str, profile: dict) -> dict:
    raw = validate_chain.invoke({**profile, "instance": instance})
    # 提取 JSON 部分（防止模型在前后加多余文字）
    start = raw.find("{")
    end = raw.rfind("}") + 1
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {"passed": False, "score": 0, "issues": ["校验结果解析失败", raw]}


if __name__ == "__main__":
    instance = generate_profile(customer_profile)
    print("【生成实例】")
    print(instance)
    print()

    validation = validate_profile(instance, customer_profile)
    print("【校验结果】")
    print(f"通过: {validation.get('passed')}  评分: {validation.get('score')}/100")
    if validation.get("issues"):
        print("问题:")
        for issue in validation["issues"]:
            print(f"  - {issue}")
    print("逐项检查:")
    for k, v in validation.get("checks", {}).items():
        status = "✓" if v else "✗"
        print(f"  {status} {k}")
