"""Build the Chinese SiguoZero architecture and training-system PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as PlatypusImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "siguozero_model_architecture_zh.pdf"
TMP = ROOT / "tmp" / "pdfs"
FORMULA_DIR = TMP / "latex-formulas-v14"
FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")

NAVY = HexColor("#12233F")
BLUE = HexColor("#2B6CB0")
CYAN = HexColor("#DDF4F7")
ORANGE = HexColor("#F6A723")
PALE_ORANGE = HexColor("#FFF1D6")
GREEN = HexColor("#248A70")
PALE_GREEN = HexColor("#E3F5EF")
RED = HexColor("#C65353")
PALE_RED = HexColor("#FBE9E7")
INK = HexColor("#1D2733")
MUTED = HexColor("#617083")
LINE = HexColor("#D6DEE8")
PAPER = HexColor("#F7F9FC")


def latex_image(
    name: str,
    expression: str,
    *,
    width: float = 440,
    font_size: float = 18,
    max_height: float = 55,
) -> PlatypusImage:
    """Render one LaTeX math expression to a high-DPI transparent image."""

    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    destination = FORMULA_DIR / f"{name}.png"
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "text.color": "#12233F",
        }
    )
    figure = plt.figure(figsize=(9.0, 0.8), dpi=360)
    figure.patch.set_alpha(0)
    figure.text(
        0.5,
        0.5,
        f"${expression}$",
        fontsize=font_size,
        color="#12233F",
        horizontalalignment="center",
        verticalalignment="center",
    )
    figure.savefig(
        destination,
        dpi=360,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    with PILImage.open(destination) as rendered:
        pixel_width, pixel_height = rendered.size
    height = width * pixel_height / pixel_width
    if height > max_height:
        width *= max_height / height
        height = max_height
    image = PlatypusImage(str(destination), width=width, height=height)
    image.hAlign = "CENTER"
    return image


def latex_box(
    name: str,
    expression: str,
    *,
    width: float = 440,
    font_size: float = 18,
    max_height: float = 55,
    fill=PAPER,
) -> Table:
    """Place a LaTeX-rendered expression in the document's formula style."""

    image = latex_image(
        name,
        expression,
        width=width,
        font_size=font_size,
        max_height=max_height,
    )
    table = Table([[image]], colWidths=[480])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei-Bold", str(FONT_BOLD)))


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleZH",
            parent=base["Title"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=27,
            leading=36,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleZH",
            parent=base["Normal"],
            fontName="MicrosoftYaHei",
            fontSize=12,
            leading=19,
            textColor=MUTED,
        ),
        "h1": ParagraphStyle(
            "H1ZH",
            parent=base["Heading1"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=19,
            leading=26,
            textColor=NAVY,
            spaceBefore=0,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2ZH",
            parent=base["Heading2"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=12,
            leading=18,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyZH",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=9.2,
            leading=15,
            textColor=INK,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "SmallZH",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=7.7,
            leading=11.5,
            textColor=INK,
        ),
        "tiny": ParagraphStyle(
            "TinyZH",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=6.8,
            leading=9.8,
            textColor=INK,
        ),
        "callout": ParagraphStyle(
            "CalloutZH",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=10,
            leading=16,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.2,
            leading=11,
            textColor=INK,
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def page_decoration(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 7 * mm, width, 7 * mm, fill=1, stroke=0)
    canvas.setFont("MicrosoftYaHei", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "SiguoZero 规则驱动自对弈系统")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.restoreState()


def add_arrow(drawing: Drawing, x1: float, y1: float, x2: float, y2: float) -> None:
    drawing.add(Line(x1, y1, x2, y2, strokeColor=BLUE, strokeWidth=1.5))
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        drawing.add(
            Polygon(
                [x2, y2, x2 - 6 * direction, y2 + 3, x2 - 6 * direction, y2 - 3],
                fillColor=BLUE,
                strokeColor=BLUE,
            )
        )
    else:
        direction = 1 if y2 > y1 else -1
        drawing.add(
            Polygon(
                [x2, y2, x2 - 3, y2 - 6 * direction, x2 + 3, y2 - 6 * direction],
                fillColor=BLUE,
                strokeColor=BLUE,
            )
        )


def add_box(
    drawing: Drawing,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: tuple[str, ...] = (),
    *,
    fill=colors.white,
    stroke=LINE,
    title_color=NAVY,
) -> None:
    drawing.add(
        Rect(
            x,
            y,
            width,
            height,
            rx=7,
            ry=7,
            fillColor=fill,
            strokeColor=stroke,
            strokeWidth=1.1,
        )
    )
    drawing.add(
        String(
            x + width / 2,
            y + height - 17,
            title,
            fontName="MicrosoftYaHei-Bold",
            fontSize=9.2,
            fillColor=title_color,
            textAnchor="middle",
        )
    )
    line_y = y + height - 34
    for line in lines:
        drawing.add(
            String(
                x + width / 2,
                line_y,
                line,
                fontName="MicrosoftYaHei",
                fontSize=7.1,
                fillColor=INK,
                textAnchor="middle",
            )
        )
        line_y -= 11


def overview_diagram() -> Drawing:
    d = Drawing(480, 220)
    add_box(d, 8, 125, 98, 70, "可执行规则环境", ("合法动作 Hard Mask", "终局 +1 / 0 / -1", "60 步无交互判和"), fill=CYAN)
    add_box(d, 145, 125, 98, 70, "布局模型", ("25 步 Pointer", "T=0.7 概率采样", "合法库存与位置"), fill=PALE_ORANGE)
    add_box(d, 282, 125, 98, 70, "策略模型", ("棋盘+阵亡先验", "1001-token 历史", "起点 -> 条件终点"), fill=PALE_GREEN)
    add_box(d, 388, 24, 84, 70, "原子检查点", ("模型 + 优化器", "中盘 + 知识", "全部 RNG"), fill=PALE_RED)
    add_box(d, 145, 24, 98, 70, "共享 current 冻结", ("全座位同一实例", "4 根 x 每根 2 次", "后缀概率采样"), fill=CYAN)
    add_box(d, 282, 24, 98, 70, "Game-GRPO", ("两副本先平均", "四候选标准化", "仅根动作回传"), fill=PALE_GREEN)
    add_arrow(d, 106, 160, 145, 160)
    add_arrow(d, 243, 160, 282, 160)
    add_arrow(d, 331, 125, 331, 94)
    add_arrow(d, 282, 59, 243, 59)
    add_arrow(d, 194, 94, 194, 125)
    add_arrow(d, 380, 59, 388, 59)
    return d


def layout_diagram() -> Drawing:
    d = Drawing(480, 250)
    add_box(d, 8, 165, 92, 60, "固定棋子序列", ("军旗 -> 3 地雷", "2 炸弹 -> 各级子力"), fill=PALE_ORANGE)
    add_box(d, 130, 165, 102, 60, "部分布局编码", ("25 个位置 token", "位置/行/类型/占用"), fill=CYAN)
    add_box(d, 262, 165, 95, 60, "双向 Transformer", ("d=256, 8 层", "8 heads, FFN=1024"), fill=PALE_GREEN)
    add_box(d, 387, 165, 85, 60, "Pointer logits", ("Query: 当前棋子", "Key: 25 点表示"), fill=PALE_ORANGE)
    add_arrow(d, 100, 195, 130, 195)
    add_arrow(d, 232, 195, 262, 195)
    add_arrow(d, 357, 195, 387, 195)
    add_box(d, 30, 52, 120, 68, "Hard Mask", ("空位", "军旗仅大本营", "地雷后两排", "炸弹不在第一排"), fill=PALE_RED)
    add_box(d, 180, 52, 120, 68, "Masked categorical", ("默认温度 T=0.7", "禁止 argmax", "记录 old log-prob"), fill=CYAN)
    add_box(d, 330, 52, 120, 68, "完整合法布局", ("25 枚、库存精确", "PlayerSetup 二次校验", "终局结果才回传"), fill=PALE_GREEN)
    add_arrow(d, 430, 165, 270, 120)
    add_arrow(d, 150, 86, 180, 86)
    add_arrow(d, 300, 86, 330, 86)
    d.add(String(240, 19, "重复 25 次：选择位置 -> 更新部分布局 -> 下一枚固定棋子", fontName="MicrosoftYaHei-Bold", fontSize=8.5, fillColor=NAVY, textAnchor="middle"))
    return d


def token_diagram() -> Drawing:
    d = Drawing(480, 175)
    add_box(d, 8, 97, 100, 55, "初始状态 token", ("动作半区全零", "棋盘+阵亡状态 256"), fill=CYAN)
    add_box(d, 138, 97, 100, 55, "转移 token", ("动作嵌入 256 维", "动作后状态 256 维"), fill=PALE_GREEN)
    add_box(d, 268, 97, 100, 55, "滑动历史", ("永久保留 T0", "最近最多 1000 步"), fill=PALE_ORANGE)
    add_box(d, 398, 97, 74, 55, "最长输入", ("1001", "x 512 维"), fill=PALE_RED)
    add_arrow(d, 108, 124, 138, 124)
    add_arrow(d, 238, 124, 268, 124)
    add_arrow(d, 368, 124, 398, 124)
    d.add(Rect(18, 20, 444, 45, rx=6, ry=6, fillColor=PAPER, strokeColor=LINE))
    d.add(String(240, 45, "未知身份=1/2/3；确定身份按座位分块=30..41 / 62..73 / 94..105 / 126..137", fontName="MicrosoftYaHei", fontSize=6.5, fillColor=INK, textAnchor="middle"))
    d.add(String(240, 30, "确定阵亡：每名对手 25 位；四暗 75 / 双明 50 / 二人 25；与布局棋子顺序一致", fontName="MicrosoftYaHei", fontSize=6.5, fillColor=INK, textAnchor="middle"))
    return d


def knowledge_diagram() -> Drawing:
    d = Drawing(480, 200)
    add_box(d, 8, 125, 100, 58, "观察者可见事实", ("已知存活身份", "公开战果/亮旗", "路径与布阵限制"), fill=CYAN)
    add_box(d, 138, 125, 100, 58, "库存约束", ("固定 25 子", "确定死亡计数", "确定存活位置"), fill=PALE_ORANGE)
    add_box(d, 268, 125, 100, 58, "枚举棋种二元组", ("逐个观察者计算", "筛选合法战果", "不读取裁判底牌"), fill=PALE_GREEN)
    add_box(d, 398, 125, 74, 58, "唯一性门", ("所有组合", "是否一致"), fill=PALE_RED)
    add_box(d, 70, 35, 140, 52, "唯一：持久写入", ("存活 -> 精确棋盘码", "死亡 -> 25 位规范槽"), fill=PALE_GREEN)
    add_box(d, 270, 35, 140, 52, "不唯一：保持未知", ("不置阵亡位", "不改变暗子棋盘码"), fill=PALE_RED)
    add_arrow(d, 108, 154, 138, 154)
    add_arrow(d, 238, 154, 268, 154)
    add_arrow(d, 368, 154, 398, 154)
    add_arrow(d, 435, 125, 175, 87)
    add_arrow(d, 435, 125, 340, 87)
    d.add(String(240, 13, "双明：自己与对家盟友共享确定知识；四暗：私有结论严格隔离", fontName="MicrosoftYaHei-Bold", fontSize=7.8, fillColor=NAVY, textAnchor="middle"))
    return d


def policy_diagram() -> Drawing:
    d = Drawing(480, 260)
    add_box(d, 8, 176, 88, 62, "棋盘整数链", ("四国 129 点", "二人 60 点", "玩家可见信息"), fill=CYAN)
    add_box(d, 120, 176, 104, 62, "Graph BoardEncoder", ("道路/铁路关系偏置", "逐点表示 + pooling", "h_board: 256"), fill=PALE_GREEN)
    add_box(d, 8, 84, 88, 62, "确定阵亡先验", ("规范 75 位", "双明/二人补零", "只含唯一结论"), fill=PALE_RED)
    add_box(d, 120, 84, 104, 62, "Casualty Projection", ("75 -> 256", "固定 25 槽/玩家", "布局顺序复用"), fill=PALE_ORANGE)
    add_box(d, 250, 151, 100, 72, "棋盘状态融合", ("concat 两个 256", "Linear + SiLU + LN", "输出 g_t: 256"), fill=CYAN)
    add_box(d, 250, 58, 100, 62, "ActionEncoder", ("起终点/行动者", "战斗/揭旗/淘汰", "输出 q_t: 256"), fill=PALE_ORANGE)
    add_box(d, 380, 151, 92, 72, "时序 Transformer", ("[q_t ; g_t]", "d=512, causal", "rollout 增量 KV"), fill=PALE_GREEN)
    add_box(d, 380, 58, 92, 62, "两阶段动作头", ("起点概率", "条件终点概率", "动态合法 mask"), fill=PALE_RED)
    add_arrow(d, 96, 207, 120, 207)
    add_arrow(d, 224, 207, 250, 188)
    add_arrow(d, 96, 115, 120, 115)
    add_arrow(d, 224, 115, 250, 171)
    add_arrow(d, 350, 187, 380, 187)
    add_arrow(d, 350, 89, 380, 165)
    add_arrow(d, 426, 151, 426, 120)
    d.add(String(240, 22, "阵亡先验融合不增加序列长度；动作概率仍由起点与条件终点分布组成", fontName="MicrosoftYaHei-Bold", fontSize=8.0, fillColor=NAVY, textAnchor="middle"))
    return d


def grpo_diagram() -> Drawing:
    d = Drawing(480, 245)
    add_box(d, 194, 193, 92, 44, "锚点 I_t", ("基础对局每一步",), fill=CYAN)
    xs = (14, 130, 246, 362)
    for index, x in enumerate(xs, start=1):
        add_box(d, x, 127, 104, 44, f"根路径 {index}", ("按冻结旧策略采样",), fill=PALE_ORANGE)
        add_arrow(d, 240, 193, x + 52, 171)
        add_box(d, x, 66, 104, 38, "2 个终局副本", ("后缀仍按旧策略采样",), fill=PALE_GREEN)
        add_arrow(d, x + 52, 127, x + 52, 104)
    add_box(d, 104, 7, 118, 42, "每根两结果平均", ("得到候选回报 Q",), fill=CYAN)
    add_box(d, 258, 7, 118, 42, "四槽组内标准化", ("得到相对优势 A",), fill=PALE_RED)
    for x in xs:
        add_arrow(d, x + 52, 66, 163, 49)
    add_arrow(d, 222, 28, 258, 28)
    return d


def training_loop_diagram() -> Drawing:
    d = Drawing(480, 230)
    add_box(d, 8, 145, 86, 58, "共享 current", ("1 Policy + 1 Layout", "全座位唯一实例"), fill=PALE_GREEN)
    add_box(d, 122, 145, 86, 58, "采样阶段冻结", ("同一 current 对象", "保存 old log-prob"), fill=CYAN)
    add_box(d, 236, 145, 96, 58, "GPU Actor 批处理", ("8 锚点有界波", "COW 历史 + 增量 KV"), fill=PALE_ORANGE)
    add_box(d, 360, 145, 112, 58, "终局完整样本", ("Policy 根组", "Layout 轨迹", "+1 / 0 / -1"), fill=PALE_RED)
    add_box(d, 236, 50, 96, 58, "双 Learner", ("GRPO + KL - entropy", "AdamW + grad clip"), fill=PALE_GREEN)
    add_box(d, 8, 50, 86, 58, "单份 reference", ("仅作 KL", "不控制任何座位"), fill=CYAN)
    add_box(d, 360, 50, 112, 58, "原子持久化", ("update 0 即保存", "latest 每 10 updates", "异常/信号紧急保存"), fill=PALE_ORANGE)
    add_arrow(d, 94, 174, 122, 174)
    add_arrow(d, 208, 174, 236, 174)
    add_arrow(d, 332, 174, 360, 174)
    add_arrow(d, 416, 145, 300, 108)
    add_arrow(d, 236, 79, 94, 79)
    add_arrow(d, 284, 108, 51, 145)
    add_arrow(d, 332, 79, 360, 79)
    d.add(String(240, 20, "恢复：模型/reference/优化器/中盘/逐玩家存活身份+阵亡库存/1000 步历史/全部 RNG", fontName="MicrosoftYaHei", fontSize=6.8, fillColor=INK, textAnchor="middle"))
    return d


def make_table(data, widths, text_style, *, header=True, font_size=7.5) -> Table:
    header_style = ParagraphStyle(
        f"{text_style.name}Header",
        parent=text_style,
        fontName="MicrosoftYaHei-Bold",
        textColor=colors.white,
    )
    converted = []
    for row_index, row in enumerate(data):
        row_style = header_style if header and row_index == 0 else text_style
        converted.append(
            [
                cell
                if hasattr(cell, "wrap")
                else Paragraph(str(cell), row_style)
                for cell in row
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 0), (-1, -1), "MicrosoftYaHei"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, PAPER]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "MicrosoftYaHei-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def callout(text: str, style: ParagraphStyle, fill=CYAN) -> Table:
    table = Table([[Paragraph(text, style)]], colWidths=[480])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 1, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def build_story(s: dict[str, ParagraphStyle]) -> list:
    story: list = []

    story.extend(
        [
            Spacer(1, 25 * mm),
            Paragraph("SiguoZero", s["title"]),
            Paragraph("四暗 / 双明 / 二人军棋", s["h1"]),
            Paragraph("Transformer 双模型自对弈强化学习系统", s["title"]),
            Spacer(1, 6 * mm),
            callout(
                "只使用可执行军棋规则、玩家可见信息和最终胜负；不使用人类棋谱、专家标签、价值塑形或显式 belief head。",
                s["callout"],
            ),
            Spacer(1, 15 * mm),
            make_table(
                [
                    ["核心约束", "已采用的实现"],
                    ["动作路径", "起点编码与终点编码的有向二元组"],
                    ["玩家模型实例", "每个模式仅 1 Policy + 1 Layout；二人/四国全部座位共享，绝不按座位复制"],
                    ["策略采样", "冻结旧策略选 4 个根路径；每根 2 次终局模拟；所有后缀继续按旧策略采样"],
                    ["时序输入", "初始棋盘 token 永久保留 + 最近最多 1000 个动作后棋盘 token"],
                    ["回报", "只在终局返回本队 +1、和棋 0、失败 -1；连续 60 步无子力交互为和棋"],
                    ["确定知识", "候选组合唯一性判定；存活身份写棋盘；阵亡按每名对手 25 位记录"],
                    ["可靠恢复", "update 0 起原子保存；中盘、逐玩家身份/阵亡先验、历史、优化器及 RNG 均可恢复"],
                ],
                [105, 375],
                s["small"],
            ),
            Spacer(1, 11 * mm),
            Paragraph("架构说明书 v1.4 · 2026-09-03", s["subtitle"]),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("1. 系统边界与三种训练模式", s["h1"]),
            overview_diagram(),
            Spacer(1, 4 * mm),
            make_table(
                [
                    ["模式", "人数 / 棋盘", "主视角可见身份", "独立运行目录"],
                    ["四暗 four_dark", "4 人 / 129 点", "己方精确；其余未确定时为 1/2/3", "runs/four_dark"],
                    ["双明 double_open", "4 人 / 129 点", "己方精确；对家为 94..105；敌方未定为 1/3", "runs/double_open"],
                    ["二人 two_player", "2 人 / 60 点", "己方 30..41；对手未定为 2，确定后 94..105", "runs/two_player"],
                ],
                [100, 90, 195, 95],
                s["small"],
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "三种模式共享同一套经过测试的组件，但训练入口、日志、检查点、优化器和 reference 模型完全分离。每个模式只创建一个当前 Policy 和一个当前 Layout，所有座位仅维护各自主视角状态并调用同一对象；参数量和玩家显存绝不乘以人数。训练另有一份只用于 KL 的 reference 模型对，它不控制玩家。加载检查点时校验模式，禁止跨模式误用。",
                s["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("2. 布局模型：棋子条件 Pointer Decoder", s["h1"]),
            layout_diagram(),
            Paragraph("固定库存与顺序", s["h2"]),
            Paragraph(
                "25 步固定序列为：军旗；地雷 x3；炸弹 x2；司令 x1；军长 x1；师长/旅长/团长/营长各 x2；连长/排长/工兵各 x3。固定棋子、只预测位置，使数量约束天然成立。",
                s["body"],
            ),
            make_table(
                [
                    ["模块", "bootstrap 设置", "输出 / 约束"],
                    ["位置 token", "d_model=256", "位置、行、是否大本营、当前占用、玩法模式"],
                    ["布局 Transformer", "8 层、8 heads、SwiGLU FFN=1024", "25 点双向上下文 + pooled layout token"],
                    ["Pointer query", "当前棋子 + 步号 + 模式 + pooled 状态", "对 25 个位置产生 logits"],
                    ["生成", "temperature=0.7 categorical", "Hard Mask 后采样并保存每步 old log-prob"],
                ],
                [105, 175, 200],
                s["small"],
            ),
            Spacer(1, 3 * mm),
            callout("布局轨迹在游戏结束前不得获得胜负梯度；完整布局始终再由 PlayerSetup 权威校验。", s["callout"], PALE_ORANGE),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("3. 策略输入：动作与完整棋盘共同构成 token", s["h1"]),
            token_diagram(),
            latex_box(
                "state_tokens",
                r"T_0=\left[\mathbf{0}_{256}\,;\,g_0\right],\qquad "
                r"T_t=\left[q_t\,;\,g_t\right]\in\mathbb{R}^{512}",
                font_size=20,
                max_height=34,
            ),
            Spacer(1, 2 * mm),
            Paragraph("棋盘整数码是类别 ID，不是连续军阶数值", s["h2"]),
            make_table(
                [
                    ["范围", "含义", "示例"],
                    ["0", "空点", "0"],
                    ["1 / 2 / 3", "四国主视角的上家 / 对家 / 下家暗子；二人只用 2", "非己方未知身份"],
                    ["30..41", "我方精确身份", "30 军旗，31 炸弹，32 工兵，...，40 司令，41 地雷"],
                    ["62..73", "上家已经确定的精确身份", "64 工兵，72 司令，73 地雷"],
                    ["94..105", "对家已经确定的精确身份；含双明盟友及二人对手", "96 工兵，104 司令，105 地雷"],
                    ["126..137", "下家已经确定的精确身份", "128 工兵，136 司令，137 地雷"],
                ],
                [80, 195, 205],
                s["small"],
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "精确码公式为 base(piece)+32 x relative-seat-block。每名其他玩家另有与布局生成顺序一致的 25 位确定阵亡行：四暗 75 位，双明只含两敌 50 位并由盟友共享，二人 25 位。候选战斗组合只有在全部合法解释一致时才写入身份/阵亡结论；不读取裁判暗子筛选，非唯一猜测保持未知。",
                s["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("3.1 确定阵亡先验与无需猜测的死规则", s["h1"]),
            knowledge_diagram(),
            latex_box(
                "casualty_prior_shape",
                r"D_t^{i,p}\in\{0,1\}^{25},\qquad "
                r"\dim(D_t^i)=75\ (\mathrm{four\_dark}),\ "
                r"50\ (\mathrm{double\_open}),\ 25\ (\mathrm{two\_player})",
                font_size=18,
                max_height=48,
            ),
            Spacer(1, 2 * mm),
            make_table(
                [
                    ["公开条件", "唯一结论"],
                    ["司令死亡", "军旗亮出；司令死亡槽置 1"],
                    ["工兵进攻存活且未夺旗", "守方死亡子只能是地雷"],
                    ["双方移除且仅一方新亮旗", "新亮旗方司令，另一方炸弹；双方新亮旗则双方司令"],
                    ["亮旗方存活棋子吃已知师长", "其司令已死，进攻子只能是军长"],
                    ["已知普通军阶同归，守子未移动且在本阵首排", "首排排除炸弹，守子只能是同军阶"],
                    ["军旗被扛 / 玩家公开出局", "该方规范 25 位全部置 1"],
                ],
                [180, 300],
                s["small"],
            ),
            Spacer(1, 3 * mm),
            callout(
                "同归仍可能有多个棋种解释时不置位；确定阵亡只记录先验事实，不记录概率推断。",
                s["callout"],
                PALE_ORANGE,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("4. 策略模型：图棋盘编码 + 因果时序 Transformer", s["h1"]),
            policy_diagram(),
            latex_box(
                "board_casualty_fusion",
                r"h_t^D=W_D\widetilde D_t,\qquad "
                r"g_t=\operatorname{LN}\!\left(\operatorname{SiLU}\!\left("
                r"W_G[h_t^B;h_t^D]+b_G\right)\right)\in\mathbb{R}^{256}",
                font_size=16,
                max_height=38,
                fill=PALE_GREEN,
            ),
            Spacer(1, 2 * mm),
            latex_box(
                "action_factorization",
                r"\log \pi_\theta(a\mid I)="
                r"\log \pi_\theta(s^{\mathrm{src}}\mid I)+"
                r"\log \pi_\theta(s^{\mathrm{dst}}\mid s^{\mathrm{src}},I)",
                font_size=18,
                max_height=34,
            ),
            Spacer(1, 2 * mm),
            make_table(
                [
                    ["层级", "bootstrap", "main", "extended"],
                    ["BoardEncoder", "4 层, d=256", "8 层, d=256", "12 层, d=256"],
                    ["时序 Transformer", "8 层, d=512", "32 层, d=512", "48 层, d=512"],
                    ["Layout", "8 层, d=256", "16 层, d=256", "24 层, d=256"],
                    ["Policy 参数", "26,610,696", "144,157,704", "215,534,600"],
                    ["Layout 参数", "8,887,296", "17,292,288", "25,697,280"],
                    ["模型对合计", "35,497,992", "161,449,992", "241,231,880"],
                ],
                [130, 116, 116, 116],
                s["small"],
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "75 位阵亡投影与棋盘融合只增加 151,040 个 Policy 参数，不改变 512 维时序宽度。所有动作都经动态合法 mask 后概率采样；表中参数只计一份共享 current，不按四个座位乘四。",
                s["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("5. Game-GRPO：4 条根路径，每条模拟 2 次", s["h1"]),
            grpo_diagram(),
            callout("每个锚点固定 4 × 2 = 8 条终局续局；不穷举合法动作，也不递归展开整棵搜索树。", s["callout"], PALE_GREEN),
            Paragraph("终局回报与优势", s["h2"]),
            latex_box(
                "grpo_return_statistics",
                r"Q_{b,k}=\frac{R_{b,k,1}+R_{b,k,2}}{2},\quad "
                r"\mu_b=\frac{1}{4}\sum_{j=1}^{4}Q_{b,j},\quad "
                r"\sigma_b=\sqrt{\frac{1}{4}\sum_{j=1}^{4}(Q_{b,j}-\mu_b)^2}",
                font_size=16,
                max_height=38,
            ),
            Spacer(1, 2 * mm),
            latex_box(
                "grpo_advantage",
                r"A_{b,k}=\frac{Q_{b,k}-\mu_b}{\sigma_b+\varepsilon_A},"
                r"\qquad \varepsilon_A=10^{-4}",
                font_size=20,
                max_height=36,
                fill=PALE_GREEN,
            ),
            Spacer(1, 2 * mm),
            Paragraph(
                "每个终局只返回本队胜 1、和 0、负 -1。同一根动作的两个结果先平均，再在四个候选槽内标准化；若四个候选回报完全相同，则该锚点优势置零，不添加中间奖励。",
                s["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("5.1 Game-GRPO 与双模型损失（LaTeX 排版）", s["h1"]),
            Paragraph("行为概率比", s["h2"]),
            latex_box(
                "importance_ratio",
                r"\rho_{b,k}(\theta)=\exp\!\left["
                r"\ell_\theta(I_b,a_{b,k})-\ell_{\mathrm{old}}(I_b,a_{b,k})"
                r"\right]",
                font_size=20,
                max_height=40,
            ),
            Paragraph("根动作裁剪目标", s["h2"]),
            latex_box(
                "policy_grpo_loss",
                r"\mathcal{L}_{P,\mathrm{GRPO}}=-\frac{1}{4B}"
                r"\sum_{b=1}^{B}\sum_{k=1}^{4}\min\!\left("
                r"\rho_{b,k}A_{b,k},\;"
                r"\mathrm{clip}(\rho_{b,k},1-\epsilon,1+\epsilon)A_{b,k}"
                r"\right)",
                font_size=15,
                max_height=48,
                fill=PALE_GREEN,
            ),
            Paragraph("带阶段参考与熵正则的完整策略损失", s["h2"]),
            latex_box(
                "total_policy_loss",
                r"\mathcal{L}_{P}=\mathcal{L}_{P,\mathrm{GRPO}}+"
                r"\beta_P D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{P,\mathrm{ref}})-"
                r"\alpha_P\mathcal{H}(\pi_\theta)",
                font_size=18,
                max_height=42,
            ),
            Paragraph("布局 Pointer 的终局损失", s["h2"]),
            latex_box(
                "layout_grpo_loss",
                r"\mathcal{L}_{L,\mathrm{GRPO}}=-\frac{1}{25BG_L}"
                r"\sum_{b,k,t}\min\!\left(\rho^L_{b,k,t}A^L_{b,k},\;"
                r"\mathrm{clip}(\rho^L_{b,k,t},1-\epsilon_L,1+\epsilon_L)A^L_{b,k}\right)",
                font_size=15,
                max_height=48,
                fill=PALE_ORANGE,
            ),
            Spacer(1, 3 * mm),
            Paragraph(
                "候选槽已经按旧策略有放回采样，损失中不能再次乘旧策略概率。策略优势只回传根动作；后缀动作仅负责把分支采样到终局。布局的 25 个 Pointer 决策共享对应座位的终局优势，但不使用 25 个概率比率的乘积。",
                s["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("6. 训练闭环、日志与可恢复性", s["h1"]),
            training_loop_diagram(),
            make_table(
                [
                    ["训练项", "默认设置"],
                    ["全局 Policy batch", "128 个完整锚点组 = 512 个根回报 = 1024 条终局续局"],
                    ["优化", "AdamW; betas=(0.9,0.95); weight decay=0.05; grad clip=1.0; 最多 3 epochs"],
                    ["学习率", "Policy 1e-4 起，2000 updates warmup 后余弦至 1e-6；KL 超限自动减半；Layout 5e-5"],
                    ["正则", "GRPO clip=0.2；KL beta=0.02，目标 0.015/action；entropy alpha=0.01"],
                    ["混合精度", "CUDA BF16，BF16 不可用时 FP16 + GradScaler；TF32 开启；归约 FP32；activation checkpoint 开启"],
                    ["详细日志", "JSONL + latest JSON + train.log + 可选 TensorBoard；loss/KL/熵/clip/胜和负/吞吐/显存"],
                ],
                [125, 355],
                s["small"],
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "安全策略是 fail-closed：非空目录没有 latest.pt 时拒绝随机重启；--no-resume 只允许空目录。正常启动自动加载 latest，不需要额外 --resume。",
                s["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("7. CUDA 容量实测与训练预算", s["h1"]),
            Paragraph(
                "实测平台：WSL2 Ubuntu 24.04，RTX 4090 24GB，PyTorch 2.11.0+cu130。共享 current Policy/Layout 各一份及 KL reference 各一份常驻；没有四份座位模型。rev.15 的页式 rollout cache 只保存冻结前缀，进入 learner 前清空。",
                s["body"],
            ),
            make_table(
                [
                    ["模型", "microbatch", "单步时间", "峰值分配", "24GB 默认"],
                    ["bootstrap", "1 / 8 / 16 / 24", "0.59 / 2.57 / 4.98 / 7.06 s", "1.71 / 7.09 / 13.79 / 20.48 GiB", "16"],
                    ["main 二人", "8 / 16 / 24", "3.54 / 6.55 / 9.23 s", "7.59 / 13.53 / 19.46 GiB", "24*"],
                    ["extended", "1 / 8", "1.22 / 6.49 s", "5.32 / 17.72 GiB", "6"],
                ],
                [85, 105, 125, 125, 40],
                s["tiny"],
            ),
            Paragraph("累计样本预算", s["h2"]),
            make_table(
                [
                    ["阶段", "learner updates", "锚点", "终局续局", "定位"],
                    ["冷启动验收", "约 7.8K", "1.0M", "8M", "验证规则、信息隔离、损失与吞吐"],
                    ["主训练", "约 62.5K", "8.0M", "64M", "建立历史回归评测和稳定棋力"],
                    ["顶尖容量预留", "200K-600K", "25.6M-76.8M", "0.205B-0.614B", "容量上限，不是棋力保证"],
                ],
                [95, 90, 85, 105, 105],
                s["small"],
            ),
            Paragraph("分模式 H100 等效规划", s["h2"]),
            make_table(
                [
                    ["模式", "冷启动", "主训练", "顶尖容量"],
                    ["二人", "约 2-3 GPU-days", "约 9-14", "约 29-128"],
                    ["四暗", "约 9-13 GPU-days", "约 58-87", "约 185-834"],
                    ["双明", "先按四暗同档", "训练后按真实局长修正", "训练后按真实局长修正"],
                ],
                [120, 120, 120, 120],
                s["small"],
            ),
            Spacer(1, 3 * mm),
            Paragraph(
                "* 二人 main 正式任务先请求 microbatch=24；真实长局或分配碎片 OOM 时，单卡 trainer 会在尚未 step 的原子 epoch 内自动降为 12。H100/H200 microbatch 与墙钟时间必须在目标集群实测。",
                s["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("8. 二人/双明 RTX 4090 实测与 H100 microbatch", s["h1"]),
            Paragraph(
                "以下 learner 容量探针条件为最长 1001-token、BF16、activation checkpoint、共享 current 与 KL reference 常驻。当前 main Policy 为 144,157,704 参数；正式长跑按 revision 15 请求二人 microbatch=24，并保留自动降到 12 的原子 OOM 回退。时间不含规则 Actor。",
                s["body"],
            ),
            make_table(
                [
                    ["模式", "RTX 4090 microbatch", "单步实测", "峰值分配"],
                    ["二人", "8", "3.54 s", "7.59 GiB"],
                    ["二人", "16", "6.55 s", "13.53 GiB"],
                    ["二人", "24", "9.23 s", "19.46 GiB"],
                    ["双明", "8", "5.27 s", "12.68 GiB"],
                    ["双明", "12", "7.46 s", "18.20 GiB"],
                ],
                [120, 140, 110, 110],
                s["small"],
            ),
            Paragraph("H100 80GB 的 1001-token 起测值", s["h2"]),
            make_table(
                [
                    ["模型规模", "二人保守", "二人向上探测", "双明保守", "全局 128 的累积步"],
                    ["bootstrap", "64", "96", "64", "2"],
                    ["main", "32", "64", "32", "4"],
                    ["extended", "24", "32", "24", "6（末批 8）"],
                ],
                [100, 85, 105, 85, 105],
                s["small"],
            ),
            Spacer(1, 3 * mm),
            latex_box(
                "gradient_accumulation",
                r"S_{\mathrm{acc}}=\left\lceil\frac{B_{\mathrm{anchor}}}{B_\mu}\right\rceil,"
                r"\qquad B_{\mathrm{anchor}}=128",
                font_size=20,
                max_height=36,
            ),
            Spacer(1, 3 * mm),
            Paragraph(
                "microbatch 的单位是完整锚点组，不是终局 rollout。H100 数字是基于 80GB 容量和本机 4090 曲线给出的起测值，并非 H100 实测保证；启动时应从保守列开始，保留至少 10% 显存，再逐级 OOM 探测。二人棋盘只有 60 点，故单独列出可尝试的上探值；双明仍是 129 点。",
                s["body"],
            ),
            callout(
                "H100 main 默认先用 microbatch=32；二人模式验证稳定后可试 48/64，双明不要未经实测直接超过 32。",
                s["callout"],
                PALE_ORANGE,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("9. rev.15 实测吞吐与 30 亿步时间", s["h1"]),
            Paragraph(
                "rev.15 从 update 10 checkpoint 完整实测 main：1,024 条 continuation 共 313,935 步；rollout 481.52 s（651.97 步/s），完整 update 488.86 s，普通原子 latest 保存约 6-8 s。microbatch=24、actor batch=192 均未回退，CUDA 峰值分配 16.66 GiB。相对旧配置 update 3-9 加权 481.37 步/s，吞吐提高 35.4%。",
                s["body"],
            ),
            Paragraph(
                "当前正式训练只运行 144.16M 参数 main，不运行 bootstrap。计入每轮保存后的单卡有效吞吐约 633 步/s，剩余目标工程排期约 61-70 天；RTX PRO 6000 估算 900-1,200 步/s、约 34-48 天，目标卡仍需实测。",
                s["small"],
            ),
            latex_box(
                "wall_clock_projection",
                r"T_{\mathrm{days}}=\frac{3\times10^9-N_{\mathrm{done}}}"
                r"{v_{\mathrm{plies/s}}\,86400\,a}",
                font_size=20,
                max_height=40,
                fill=PALE_GREEN,
            ),
            Spacer(1, 3 * mm),
            make_table(
                [
                    ["硬件", "聚合吞吐", "纯 rollout", "含 learner/评测规划"],
                    ["1× RTX 4090", "633 步/s（实测外推）", "54.8 天", "61-70 天"],
                    ["2× RTX 4090", "1,076-1,139 步/s", "30.4-32.2 天", "34-41 天"],
                    ["3× RTX 4090", "1,519-1,633 步/s", "21.2-22.8 天", "24-30 天"],
                    ["4× RTX 4090", "1,823-2,026 步/s", "17.1-19.0 天", "20-26 天"],
                    ["1× RTX PRO 6000 96GB", "900-1,200 步/s", "28.9-38.5 天", "34-48 天"],
                    ["2× RTX PRO 6000 96GB", "1,530-2,160 步/s", "16.1-22.7 天", "19-29 天"],
                    ["1× H100 / B200 / B300", "850-1,650 步/s", "21-41 天", "27-55 天"],
                ],
                [145, 115, 95, 125],
                s["tiny"],
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "4090 多卡按 DDP 效率 85%-90%（2 卡）、80%-86%（3 卡）和 72%-80%（4 卡）外推。PRO/H100/B200/B300 均未在本项目实测；更大显存允许增大 actor/learner batch，但 Python 裁判和小 kernel 使速度不会按峰值 TOPS 线性增长。",
                s["body"],
            ),
            callout(
                "当前仓库已支持 torchrun/DDP：每 GPU 一个同步计算 rank，但每个 rank 内所有座位仍共享同一 Policy/Layout。表中跨卡数值仍需目标机器完整 update 复测。",
                s["callout"],
                PALE_ORANGE,
            ),
            Paragraph(
                "30 亿步是累计 continuation_plies，不是 optimizer step，也不是冠军保证。update 11 平均 306.58 步/rollout，对应约 9,555 updates；按剩余步数当前循环连续运行约 54.8 天，含可用率、分布变化和训练外评测按 61-70 天。仍须使用至少 10 个 update 的移动中位数重算。",
                s["small"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("10. 一共训练多少个 rollout", s["h1"]),
            Paragraph(
                "本文把一个 rollout 严格定义为：从某个锚点固定一个根动作后，用一个独立副本继续按旧策略采样直到规则终局。每个锚点有 4 个根候选、每候选 2 个副本，因此产生 8 个 rollout。基础对局本身和训练外评测局不计入此数。",
                s["body"],
            ),
            latex_box(
                "rollout_count",
                r"N_{\mathrm{rollout}}=U\,B_{\mathrm{anchor}}KM="
                r"U\times128\times4\times2=1024U",
                font_size=21,
                max_height=42,
                fill=PALE_GREEN,
            ),
            Spacer(1, 4 * mm),
            make_table(
                [
                    ["累计阶段", "learner updates", "每模式 rollout", "二人 + 双明合计", "两模式分叉环境步"],
                    ["冷启动验收", "约 7.8K", "8.0M", "16.0M", "约 4.14B"],
                    ["有竞争力主训练", "62.5K", "64.0M", "128.0M", "约 33.09B"],
                    ["顶尖容量预留", "200K-600K", "204.8M-614.4M", "409.6M-1.2288B", "约 105.9B-317.6B"],
                ],
                [100, 90, 100, 100, 90],
                s["tiny"],
            ),
            Paragraph("推荐的明确口径", s["h2"]),
            make_table(
                [
                    ["问题", "答案"],
                    ["每次 update", "每个模式 1,024 个终局 rollout"],
                    ["二人 30 亿步", "首轮折算 9.37M rollouts / 9.15K updates；规划范围仍为 8.6M-16.4M / 8.4K-16.1K"],
                    ["先完成可用主训练", "每模式累计 64M；二人 + 双明一共 128M"],
                    ["冲击顶尖棋力", "每模式累计 204.8M-614.4M；两模式合计 409.6M-1.2288B"],
                    ["是否保证顶尖", "不保证；每 10K-25K updates 依据历史回归、人类盲测和熵决定继续或停止"],
                ],
                [130, 350],
                s["small"],
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "顶尖容量是预算上限而非必须烧完。由于所有座位共享同一个模型，rollout 数不会再乘玩家数；四国四个座位只改变每条续局中的观测与行动轮转，不把 8 条续局变成 32 条。",
                s["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("11. 验证结果、运行入口与生产边界", s["h1"]),
            make_table(
                [
                    ["验证", "结果"],
                    ["规则/编码/共享测试", "97 项 pytest 全部通过（另含 129 个参数化子例），覆盖确定身份/阵亡先验、60 步判和、暗子隔离、座位共享与 DDP"],
                    ["4x2 rollout", "严格生成 4 个根候选 x 2 副本，全部运行到规则终局"],
                    ["三模式 CUDA", "四暗 / 双明 / 二人分别完成训练 update、下一 update 自动恢复和检查点推理"],
                    ["恢复完整性", "格式 v4 保存布局缓冲、中盘、逐玩家身份/阵亡库存、历史、模型/优化器/reference 与各 rank RNG"],
                    ["最长上下文", "bootstrap/main/extended 均完成四暗 1001-token CUDA 前向、反向和 AdamW 容量探针"],
                ],
                [125, 355],
                s["small"],
            ),
            Paragraph("三个独立入口", s["h2"]),
            Table(
                [[Paragraph(
                    "python -m junqi.training.train_four_dark --device cuda<br/>"
                    "python -m junqi.training.train_double_open --device cuda<br/>"
                    "python -m junqi.training.train_two_player --device cuda<br/><br/>"
                    "python -m junqi.training.infer_four_dark --checkpoint runs/four_dark/checkpoints/latest.pt<br/>"
                    "python tools/benchmark_cuda.py --mode four_dark --model-scale main --context-tokens 1001 --batch-size 8<br/>"
                    "python tools/benchmark_rollout.py --mode two_player --model-scale main --anchors 8 --max-game-plies 600 --full-stack",
                    s["code"],
                )]],
                colWidths=[480],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PAPER),
                        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            ),
            Paragraph("达到大规模预算前仍需完成", s["h2"]),
            Paragraph(
                "已完成：单机 DDP、物理页式 causal KV、页级 COW、单次动作 GPU→CPU 同步、packed 棋盘、长度分桶与规则静态查表。后续：1) C++/Rust 完整批量裁判并与 Python 裁判差分；2) 融合 paged-attention/CUDA Graph，消除 page gather 和小 kernel；3) 一版本 behavior snapshot 的跨 update Actor/Learner 流水；4) 训练外历史回归、人类盲测和长期断电演练。",
                s["body"],
            ),
            Paragraph("实现与研究依据", s["h2"]),
            Paragraph(
                "代码：src/junqi/training/；配置：bootstrap.yaml rev.15；资源表：docs/training_resources_zh.md。规则依据：<link href='https://www.gameabc.com/news/201704/3333.html' color='#2B6CB0'>边锋军棋规则</link>、<link href='https://www.junqi.app/zh/rules' color='#2B6CB0'>军棋玩法指南</link>。硬件规格：<link href='https://www.nvidia.com/en-us/data-center/h100/' color='#2B6CB0'>NVIDIA H100</link>、<link href='https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html' color='#2B6CB0'>NVIDIA HGX B200</link>、<link href='https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/' color='#2B6CB0'>RTX PRO 6000</link>。跨卡时间是项目推算，不是 NVIDIA benchmark。",
                s["small"],
            ),
        ]
    )
    return story


def main() -> None:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=19 * mm,
        title="SiguoZero 模型架构与训练系统",
        author="OpenAI Codex",
        subject="四暗、双明和二人军棋的双 Transformer 自对弈系统",
    )
    doc.build(build_story(styles()), onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(OUTPUT)


if __name__ == "__main__":
    main()
