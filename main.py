#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
astrbot_plugin_tts_emotion_tags
Astar 的情绪表现标签注入插件（接管式 v2）

流水线：
  LLM 请求 → on_llm_request:  向 system prompt 追加标签使用约定（软约束）
  LLM 回复 → on_llm_response: 强制规范化标签（白名单过滤，非法丢弃）
  发送前   → on_decorating_result:
      若 enable_tts_takeover=True:
        插件自己调 TTS provider.get_audio(带标签文本) 合成语音
        替换 Plain → Record(语音) + 干净文本 Plain
        跳过 AstrBot 内置 TTS
      若 enable_tts_takeover=False:
        插件完全放手，回归 AstrBot 原生 TTS 行为
"""

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import LLMResponse
from astrbot.api import logger, AstrBotConfig
from astrbot.api.message_components import Plain, Record
import re
import traceback
import random

# ============================================================
# 表现标签白名单 & 别名归一化表
# ============================================================

SUPPORTED_TAGS = {
    "chuckle", "laughs", "breath", "inhale", "exhale", "sighs",
    "humming", "emm", "hissing", "coughs", "clearthroat", "pant",
    "gasps", "sniffs", "snorts", "lip-smacking", "sneezes",
}

_ALIAS_TAG_MAP = {
    # 中文表演词
    "轻笑": "chuckle", "笑": "laughs", "大笑": "laughs", "哈哈": "laughs",
    "呼吸": "breath", "吸气": "inhale", "呼气": "exhale",
    "叹气": "sighs", "叹口气": "sighs",
    "哼唱": "humming", "哼歌": "humming", "哼": "humming",
    "嗯": "emm", "恩": "emm",
    "哈气": "hissing", "嘶": "hissing",
    "咳嗽": "coughs", "咳": "coughs",
    "清嗓": "clearthroat", "清嗓子": "clearthroat",
    "喘气": "pant", "喘": "pant",
    "惊讶倒吸": "gasps", "倒吸": "gasps",
    "抽气": "sniffs", "抽泣": "sniffs", "吸鼻子": "sniffs",
    "轻哼": "snorts", "哼鼻": "snorts",
    "咂嘴": "lip-smacking", "吧唧": "lip-smacking",
    "打喷嚏": "sneezes", "喷嚏": "sneezes",
    # 常见英文变体（大小写/驼峰/连字符/复数）
    "chuckle": "chuckle", "chuckles": "chuckle",
    "laugh": "laughs", "laughs": "laughs", "Laughs": "laughs", "LAUGHS": "laughs",
    "breath": "breath", "Breath": "breath", "BREATH": "breath",
    "inhale": "inhale", "Inhale": "inhale",
    "exhale": "exhale", "Exhale": "exhale",
    "sigh": "sighs", "sighs": "sighs", "Sigh": "sighs", "SIGHS": "sighs",
    "humming": "humming", "Humming": "humming",
    "emm": "emm", "Emm": "emm", "umm": "emm",
    "hiss": "hissing", "hissing": "hissing", "Hissing": "hissing",
    "cough": "coughs", "coughs": "coughs",
    "clear-throat": "clearthroat", "clear_throat": "clearthroat", "clearthroat": "clearthroat",
    "pant": "pant", "panting": "pant",
    "gasp": "gasps", "gasps": "gasps", "Gasp": "gasps",
    "sniff": "sniffs", "sniffs": "sniffs",
    "snort": "snorts", "snorts": "snorts",
    "lip-smacking": "lip-smacking", "lip_smacking": "lip-smacking",
    "sneeze": "sneezes", "sneezes": "sneezes",
}


@register(
    "astrbot_plugin_tts_emotion_tags",
    "Astar",
    "LLM 回复注入 TTS 表现标签并接管 MiniMax 合成",
    "0.2.0",
    "",
)
class TTSEmotionTagsPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

    # ============================================================
    # 钩子1: LLM 请求 — 注入标签使用约定（软约束）
    # ============================================================
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        if not self.config.get("enable_tts_takeover", True):
            return  # 接管关闭时不注入

        convention = (
            "\n\n[语音表现标签约定]\n"
            "当你的发言需要表达情绪/动作时，可以在相应句子中插入 MiniMax 表演标签，"
            "例如 (chuckle) 轻笑、(laughs) 笑、(breath) 换气、(inhale) 吸气、"
            "(exhale) 呼气、(sighs) 叹气、(humming) 哼唱、(emm) 嗯、"
            "(coughs) 咳嗽、(pant) 喘气、(gasps) 倒吸、(sniffs) 抽气等。"
            "停顿可用 <#0.3#> 表示 0.3 秒。"
            "标签请用英文小写、半角括号，直接写在要表演的词句附近，如："
            "\"找到啦(chuckle)原来是端口没监听。\" "
            "这些标签仅用于语音合成，系统会自动从发送文本中移除，用户不会看到，请放心使用。"
        )
        sp = getattr(req, "system_prompt", "")
        if sp and convention not in sp:
            req.system_prompt = sp + convention

    # ============================================================
    # 钩子2: LLM 回复 — 强制规范化标签
    # ============================================================
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse, *args):
        if not self.config.get("enable_tts_takeover", True):
            return  # 接管关闭时不处理

        if not resp or not resp.completion_text:
            return

        original = resp.completion_text
        tagged = self._decorate_text(original)

        if tagged != original:
            resp.completion_text = tagged
            logger.info(
                f"[TTS Emotion Tags] 已规范化表现标签:\n"
                f"  原文: {original[:80]}...\n"
                f"  规范: {tagged[:100]}..."
            )

    # ============================================================
    # 钩子3: 发送前 — 接管合成 or 剥离标签
    # ============================================================
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        takeover = self.config.get("enable_tts_takeover", True)

        result = event.get_result()
        if not hasattr(result, "chain") or not result.chain:
            return

        if takeover:
            # ===== 接管模式：插件自己调 TTS，带标签合成 =====
            await self._takeover_tts(event, result)
        else:
            # ===== 非接管模式：仅剥离标签，回归 AstrBot 原生 =====
            if not self.config.get("enable_strip", True):
                return
            for comp in result.chain:
                text = getattr(comp, "text", None)
                if isinstance(text, str):
                    cleaned = self._strip_tags(text)
                    if cleaned != text:
                        comp.text = cleaned

    async def _takeover_tts(self, event: AstrMessageEvent, result):
        """接管 TTS 合成：按触发概率，将【整条回复】转成一条带表演语音 + 干净文本。

        关键点：
          - 把多个 Plain（分段回复产生）合并成一个整体，只合成【一次】语音
          - 尊重 trigger_probability：命中概率才合成语音，否则保持纯文本
          - 文本始终保留（正确处理分段，不让每段都变成语音）
        """
        try:
            tts_provider = await self.context.get_using_tts_provider_async(
                event.unified_msg_origin
            )
        except Exception:
            logger.error("[TTS Emotion Tags] 获取 TTS provider 失败，回退到纯文本")
            tts_provider = None

        try:
            prob = float(self.config.get("trigger_probability", 0.3))
            prob = max(0.0, min(prob, 1.0))
        except (TypeError, ValueError):
            prob = 0.3

        should_tts = tts_provider is not None and random.random() <= prob

        # 1) 分离出所有 Plain 组件，并把它们合并成一个整体文本
        #    非 Plain 组件（图片等）不动，保持原样
        plain_texts = []
        plain_positions = []  # 记录原始链里 Plain 的位置（用于合并后回填）
        for i, comp in enumerate(result.chain):
            if isinstance(comp, Plain) and len(comp.text) > 1:
                plain_texts.append((i, comp.text))
                plain_positions.append(i)

        if not plain_texts:
            return

        # 合并整条文本（用换行连接各分段）
        full_tts_text = "\n".join(t for _, t in plain_texts)
        full_clean_text = self._strip_tags(full_tts_text).strip()

        if not should_tts or not full_clean_text:
            # 未命中概率：只需清理各分段文本中的标签，保持原分段结构
            for i, t in plain_texts:
                result.chain[i].text = self._strip_tags(t)
            return

        # 2) 命中概率：合成一条语音
        try:
            logger.info(f"[TTS Emotion Tags] 接管合成(整体): {full_tts_text[:60]}...")
            audio_path = await tts_provider.get_audio(full_tts_text)
        except Exception:
            logger.error(f"[TTS Emotion Tags] 合成异常:\n{traceback.format_exc()}")
            audio_path = None

        if not audio_path:
            # 合成失败：回退纯文本（清理标签）
            for i, t in plain_texts:
                result.chain[i].text = self._strip_tags(t)
            return

        # 3) 构造新链：把原始 Plain 替换成 一条语音 + 干净的整条文本
        new_chain = []
        plain_set = set(plain_positions)
        for i, comp in enumerate(result.chain):
            if i in plain_set:
                if i == plain_positions[0]:
                    # 第一个 Plain 位置：放语音 + 合并后的干净文本
                    new_chain.append(Record(file=audio_path, url=audio_path, text=full_clean_text))
                    new_chain.append(Plain(full_clean_text))
                # 其他 Plain 位置跳过（已合并）
                continue
            new_chain.append(comp)

        result.chain = new_chain

    # ============================================================
    # 工具函数
    # ============================================================
    def _decorate_text(self, text: str) -> str:
        """强制规范化：识别标签→归一化；无法识别的丢弃，绝不喂给 TTS。"""
        if not text:
            return text

        def _normalize_one(match):
            raw = match.group(0)
            inner = raw[1:-1].strip()
            if not inner:
                return ""
            key = inner.lower().replace(" ", "").replace("_", "-")
            if key in _ALIAS_TAG_MAP:
                return f"({_ALIAS_TAG_MAP[key]})"
            if key in SUPPORTED_TAGS:
                return f"({key})"
            return ""  # 无法识别 → 丢弃

        pattern = r"\([^)]*\)|（[^）]*）"
        tagged = re.sub(pattern, _normalize_one, text)

        # 停顿指令规范化：<#0.35#> 统一格式
        tagged = re.sub(r"<#\s*(\d+(?:\.\d+)?)\s*>", r"<#\1#>", tagged)

        # 清理多余空格（只压缩空格，保留换行结构）
        tagged = re.sub(r"[ 	]{2,}", " ", tagged)
        return tagged.strip()

    def _strip_tags(self, text: str) -> str:
        """超宽剥离：移除一切括号标签和停顿指令，保证 QQ 文本干净。"""
        if not text:
            return text
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"（[^）]*）", "", text)
        text = re.sub(r"<#\s*\d+(?:\.\d+)?#>", "", text)
        text = re.sub(r"[ 	]{2,}", " ", text)
        return text.strip()
