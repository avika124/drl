from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import datetime

# Create PDF
pdf_path = "/sessions/confident-quirky-lovelace/mnt/drl/DRL_Node_Guide.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=28,
    textColor=HexColor('#1a3a52'),
    spaceAfter=12,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=16,
    textColor=HexColor('#2d5a7b'),
    spaceAfter=10,
    spaceBefore=12,
    fontName='Helvetica-Bold'
)

subheading_style = ParagraphStyle(
    'CustomSubHeading',
    parent=styles['Heading3'],
    fontSize=13,
    textColor=HexColor('#3d6a8b'),
    spaceAfter=6,
    spaceBefore=8,
    fontName='Helvetica-Bold'
)

normal_style = ParagraphStyle(
    'CustomNormal',
    parent=styles['Normal'],
    fontSize=10,
    alignment=TA_JUSTIFY,
    spaceAfter=8,
    leading=12
)

example_style = ParagraphStyle(
    'Example',
    parent=styles['Normal'],
    fontSize=9,
    textColor=HexColor('#555555'),
    leftIndent=20,
    spaceAfter=6,
    fontName='Courier'
)

story = []

# Title Page
story.append(Spacer(1, 1.5*inch))
story.append(Paragraph("DRL Workflow Node Guide", title_style))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("Complete Input • Output • Config Reference", styles['Heading2']))
story.append(Spacer(1, 0.5*inch))
story.append(Paragraph("Understanding the 7 Nodes in Campaign Optimization System", styles['Normal']))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
story.append(PageBreak())

# Table of Contents
story.append(Paragraph("Table of Contents", heading_style))
story.append(Spacer(1, 0.2*inch))
toc_items = [
    "Node 1: MockCampaignEnv / BigQuery",
    "Node 2: Replay Buffer",
    "Node 3: SACAgent",
    "Node 4: Checkpoint",
    "Node 5: Load SAC",
    "Node 6: SafeDRLAgent",
    "Node 7: HybridOptimizer"
]
for i, item in enumerate(toc_items, 1):
    story.append(Paragraph(f"{i}. {item}", normal_style))
story.append(Spacer(1, 0.3*inch))
story.append(PageBreak())

# NODE 1
story.append(Paragraph("Node 1: MockCampaignEnv / BigQuery", heading_style))
story.append(Paragraph("What It Does", subheading_style))
story.append(Paragraph(
    "This node <b>fetches campaign data</b> from your advertising platforms (Google Ads, Facebook, TikTok) via BigQuery, converts raw metrics into a 42-dimensional state vector, and calculates a performance reward score.",
    normal_style
))

story.append(Paragraph("INPUTS", subheading_style))
story.append(Paragraph("What data does this node need?", normal_style))
input_data_n1 = [
    ['Field', 'Type', 'Sample', 'Source'],
    ['campaign_id', 'string', 'cmp_7f3a', 'external: seed / user input'],
    ['run_id', 'string', 'run_2026_04_03', 'external: seed / user input'],
]
input_table_n1 = Table(input_data_n1, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1.5*inch])
input_table_n1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(input_table_n1)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "<b>campaign_id:</b> A unique code identifying which campaign to optimize (e.g., 'cmp_7f3a' = Summer Sale campaign on Google Search)",
    normal_style
))
story.append(Paragraph(
    "<b>run_id:</b> A unique code identifying which training/test run this is (e.g., 'run_2026_04_03' = first run on April 3, 2026)",
    normal_style
))

story.append(Paragraph("OUTPUTS", subheading_style))
story.append(Paragraph("What does this node produce?", normal_style))
output_data_n1 = [
    ['Field', 'Type', 'Sample', 'Used In'],
    ['state_vector', 'float32[42]', '[0.12, 0.04, 2.30, ...]', 'n2 Replay Buffer'],
    ['reward', 'float', '0.034', 'n2 Replay Buffer / Training'],
]
output_table_n1 = Table(output_data_n1, colWidths=[1.5*inch, 1.2*inch, 1.3*inch, 1.5*inch])
output_table_n1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(output_table_n1)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "<b>state_vector:</b> A list of 42 numbers representing the campaign's metrics. Each number is normalized between 0-1. Includes CTR, CVR, ROAS, CPA, budget metrics, audience data, and constraints.",
    normal_style
))
story.append(Paragraph(
    "<b>reward:</b> A score from 0-1 indicating how well the campaign performed. Higher = better performance. Calculated from weighted formula: (ROAS × 0.4) + (CTR × 0.2) + (CVR × 0.2) + (1/CPA × 0.2)",
    normal_style
))

story.append(Paragraph("CONFIGURATION", subheading_style))
config_data_n1 = [
    ['Parameter', 'Value', 'Type', 'Source'],
    ['state_dim', '42', 'int', 'config.yaml'],
    ['max_steps', '100', 'int', 'train.py'],
]
config_table_n1 = Table(config_data_n1, colWidths=[1.5*inch, 1*inch, 1*inch, 1.8*inch])
config_table_n1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(config_table_n1)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "<b>state_dim = 42:</b> Each state_vector must have exactly 42 numbers. These include core metrics (CTR, CVR, ROAS, CPA, CPM), volume metrics (spend, impressions), temporal metrics (time of day, seasonality), trends, competitive metrics, ML scores, and constraint features.",
    normal_style
))
story.append(Paragraph(
    "<b>max_steps = 100:</b> Maximum number of times this node runs before stopping. Higher values = more data collected = better AI training (but slower execution).",
    normal_style
))
story.append(PageBreak())

# NODE 2
story.append(Paragraph("Node 2: Replay Buffer", heading_style))
story.append(Paragraph("What It Does", subheading_style))
story.append(Paragraph(
    "This node <b>stores training data</b> in a prioritized experience replay buffer. It collects state-action-reward transitions and serves random batches to the AI trainer. Uses importance sampling to prioritize important experiences.",
    normal_style
))

story.append(Paragraph("INPUTS", subheading_style))
input_data_n2 = [
    ['Field', 'Type', 'Sample', 'Source'],
    ['state_vector', 'float32[42]', '[0.12, 0.04, ...]', 'n1 MockCampaignEnv'],
    ['reward', 'float', '0.034', 'n1 MockCampaignEnv'],
    ['action', 'float', '[0.15, 0.08, ...]', 'n3 SACAgent'],
    ['next_state', 'float32[42]', '[0.13, 0.05, ...]', 'n1 MockCampaignEnv'],
]
input_table_n2 = Table(input_data_n2, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 1.6*inch])
input_table_n2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(input_table_n2)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "This node receives complete transitions: the campaign state BEFORE an action, the action taken, the reward received, and the campaign state AFTER.",
    normal_style
))

story.append(Paragraph("OUTPUTS", subheading_style))
output_data_n2 = [
    ['Field', 'Type', 'Sample', 'Used In'],
    ['batch_states', 'float32[batch_size, 42]', 'array of 64 states', 'n3 SACAgent'],
    ['batch_actions', 'float[batch_size]', 'array of 64 actions', 'n3 SACAgent'],
    ['batch_rewards', 'float[batch_size]', 'array of 64 rewards', 'n3 SACAgent'],
    ['batch_next_states', 'float32[batch_size, 42]', 'array of 64 states', 'n3 SACAgent'],
]
output_table_n2 = Table(output_data_n2, colWidths=[1.3*inch, 1.3*inch, 1.2*inch, 1.6*inch])
output_table_n2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(output_table_n2)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "Outputs mini-batches of experiences (typically batch_size=64). Each batch contains random samples from 50,000+ stored transitions. Samples are prioritized by importance.",
    normal_style
))

story.append(Paragraph("CONFIGURATION", subheading_style))
config_data_n2 = [
    ['Parameter', 'Value', 'Type', 'Source'],
    ['buffer_size', '50000', 'int', 'config.yaml'],
    ['batch_size', '64', 'int', 'config.yaml'],
    ['alpha', '0.6', 'float', 'config.yaml'],
]
config_table_n2 = Table(config_data_n2, colWidths=[1.5*inch, 1*inch, 1*inch, 1.8*inch])
config_table_n2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(config_table_n2)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "<b>buffer_size:</b> Maximum number of transitions to store (50,000). Older data gets replaced when buffer is full.",
    normal_style
))
story.append(Paragraph(
    "<b>batch_size:</b> How many transitions per mini-batch (64). Larger batches = more stable training but slower.",
    normal_style
))
story.append(Paragraph(
    "<b>alpha:</b> Prioritization strength (0.6). Higher = prioritize important experiences more; 0 = uniform random sampling.",
    normal_style
))
story.append(PageBreak())

# NODE 3
story.append(Paragraph("Node 3: SACAgent (AI Brain)", heading_style))
story.append(Paragraph("What It Does", subheading_style))
story.append(Paragraph(
    "This is the <b>core AI learner</b> using Soft Actor-Critic (SAC) algorithm. It learns optimal bidding and budget allocation strategies by analyzing state-action-reward patterns. Trains neural networks to predict good actions.",
    normal_style
))

story.append(Paragraph("INPUTS", subheading_style))
input_data_n3 = [
    ['Field', 'Type', 'Sample', 'Source'],
    ['batch_states', 'float32[64, 42]', '64 campaign snapshots', 'n2 Replay Buffer'],
    ['batch_actions', 'float[64]', '64 historical actions', 'n2 Replay Buffer'],
    ['batch_rewards', 'float[64]', '64 reward scores', 'n2 Replay Buffer'],
    ['batch_next_states', 'float32[64, 42]', '64 next snapshots', 'n2 Replay Buffer'],
]
input_table_n3 = Table(input_data_n3, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 1.6*inch])
input_table_n3.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(input_table_n3)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "Takes mini-batches of historical campaigns and asks: 'Given this campaign state, what action was taken? Did that action lead to good rewards?' Uses this to update its neural network weights.",
    normal_style
))

story.append(Paragraph("OUTPUTS", subheading_style))
output_data_n3 = [
    ['Field', 'Type', 'Sample', 'Used In'],
    ['actor_loss', 'float', '0.0234', 'monitoring / logging'],
    ['critic_loss', 'float', '0.0145', 'monitoring / logging'],
    ['q_value', 'float', '0.567', 'decision making'],
    ['policy', 'neural network', 'trained weights', 'n5 Load SAC'],
]
output_table_n3 = Table(output_data_n3, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 1.6*inch])
output_table_n3.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(output_table_n3)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "<b>actor_loss & critic_loss:</b> Training error metrics. Lower = better learning.",
    normal_style
))
story.append(Paragraph(
    "<b>q_value:</b> Estimated quality of actions (0-1 scale). Used to rank decisions.",
    normal_style
))
story.append(Paragraph(
    "<b>policy:</b> The trained neural network that maps campaign states to optimal actions.",
    normal_style
))

story.append(Paragraph("CONFIGURATION", subheading_style))
config_data_n3 = [
    ['Parameter', 'Value', 'Type', 'Source'],
    ['learning_rate', '0.0003', 'float', 'config.yaml'],
    ['hidden_dim', '256', 'int', 'config.yaml'],
    ['gamma', '0.99', 'float', 'config.yaml'],
    ['target_entropy', '-2.0', 'float', 'config.yaml'],
]
config_table_n3 = Table(config_data_n3, colWidths=[1.5*inch, 1.2*inch, 1*inch, 1.6*inch])
config_table_n3.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(config_table_n3)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "<b>learning_rate:</b> How fast the AI adjusts its strategy (0.0003 = conservative). Lower = slower but more stable learning.",
    normal_style
))
story.append(Paragraph(
    "<b>hidden_dim:</b> Size of neural network hidden layers (256). Larger = more capacity to learn complex patterns.",
    normal_style
))
story.append(Paragraph(
    "<b>gamma:</b> Future reward discount factor (0.99). How much weight to give future rewards vs immediate rewards.",
    normal_style
))
story.append(PageBreak())

# NODE 4
story.append(Paragraph("Node 4: Checkpoint (Save Model)", heading_style))
story.append(Paragraph("What It Does", subheading_style))
story.append(Paragraph(
    "This node <b>saves the trained AI model</b> to disk. It stores the neural network weights, architecture, and optimizer state. Acts as a checkpoint for later inference.",
    normal_style
))

story.append(Paragraph("INPUTS", subheading_style))
input_data_n4 = [
    ['Field', 'Type', 'Sample', 'Source'],
    ['policy', 'neural network', 'trained weights', 'n3 SACAgent'],
    ['optimizer_state', 'dict', 'adam state dict', 'n3 SACAgent'],
    ['episode_count', 'int', '1000', 'training loop'],
]
input_table_n4 = Table(input_data_n4, colWidths=[1.5*inch, 1.3*inch, 1.2*inch, 1.5*inch])
input_table_n4.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(input_table_n4)
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("OUTPUTS", subheading_style))
output_data_n4 = [
    ['Field', 'Type', 'Sample', 'Used In'],
    ['checkpoint_file', 'file', 'sac_model_ep1000.pt', 'n5 Load SAC'],
    ['metadata', 'dict', '{episode: 1000, loss: 0.023}', 'logging / monitoring'],
]
output_table_n4 = Table(output_data_n4, colWidths=[1.5*inch, 1.2*inch, 1.5*inch, 1.4*inch])
output_table_n4.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(output_table_n4)
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("CONFIGURATION", subheading_style))
config_data_n4 = [
    ['Parameter', 'Value', 'Type', 'Source'],
    ['save_dir', './models/', 'string', 'config.yaml'],
    ['save_interval', '100', 'int', 'config.yaml'],
]
config_table_n4 = Table(config_data_n4, colWidths=[1.5*inch, 1.5*inch, 1*inch, 1.6*inch])
config_table_n4.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(config_table_n4)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "<b>save_dir:</b> Where to save model files on disk.",
    normal_style
))
story.append(Paragraph(
    "<b>save_interval:</b> Save a checkpoint every N training episodes.",
    normal_style
))
story.append(PageBreak())

# NODE 5
story.append(Paragraph("Node 5: Load SAC (Load Trained Model)", heading_style))
story.append(Paragraph("What It Does", subheading_style))
story.append(Paragraph(
    "This node <b>loads a saved AI model</b> from disk for inference (making predictions on real campaigns). Restores all neural network weights and makes the model ready to generate recommendations.",
    normal_style
))

story.append(Paragraph("INPUTS", subheading_style))
input_data_n5 = [
    ['Field', 'Type', 'Sample', 'Source'],
    ['checkpoint_path', 'string', './models/sac_model.pt', 'file system'],
    ['campaign_state', 'float32[42]', '[0.12, 0.04, ...]', 'n1 MockCampaignEnv'],
]
input_table_n5 = Table(input_data_n5, colWidths=[1.5*inch, 1.3*inch, 1.2*inch, 1.5*inch])
input_table_n5.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(input_table_n5)
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("OUTPUTS", subheading_style))
output_data_n5 = [
    ['Field', 'Type', 'Sample', 'Used In'],
    ['loaded_policy', 'neural network', 'trained weights', 'n6 SafeDRLAgent'],
    ['model_info', 'dict', '{trained_episodes: 1000}', 'logging'],
]
output_table_n5 = Table(output_data_n5, colWidths=[1.5*inch, 1.3*inch, 1.2*inch, 1.5*inch])
output_table_n5.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(output_table_n5)
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("CONFIGURATION", subheading_style))
config_data_n5 = [
    ['Parameter', 'Value', 'Type', 'Source'],
    ['load_best', 'true', 'bool', 'config.yaml'],
    ['device', 'cuda', 'string', 'config.yaml'],
]
config_table_n5 = Table(config_data_n5, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.6*inch])
config_table_n5.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(config_table_n5)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "<b>load_best:</b> Whether to load the best-performing model checkpoint (vs latest).",
    normal_style
))
story.append(Paragraph(
    "<b>device:</b> Where to run the model (cuda=GPU for faster inference, cpu=slower but always available).",
    normal_style
))
story.append(PageBreak())

# NODE 6
story.append(Paragraph("Node 6: SafeDRLAgent (Safety Guardrails)", heading_style))
story.append(Paragraph("What It Does", subheading_style))
story.append(Paragraph(
    "This node <b>applies safety constraints</b> to AI recommendations. It prevents extreme actions (e.g., 100x bid increase) and enforces business rules. Ensures the AI doesn't make unrealistic or costly mistakes.",
    normal_style
))

story.append(Paragraph("INPUTS", subheading_style))
input_data_n6 = [
    ['Field', 'Type', 'Sample', 'Source'],
    ['raw_action', 'float', '0.85', 'n5 Load SAC'],
    ['campaign_state', 'float32[42]', '[0.12, 0.04, ...]', 'n1 MockCampaignEnv'],
    ['constraints', 'dict', '{max_bid_mult: 2.0, ...}', 'config.yaml'],
]
input_table_n6 = Table(input_data_n6, colWidths=[1.4*inch, 1.2*inch, 1.3*inch, 1.6*inch])
input_table_n6.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(input_table_n6)
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("OUTPUTS", subheading_style))
output_data_n6 = [
    ['Field', 'Type', 'Sample', 'Used In'],
    ['safe_action', 'float', '0.52', 'n7 HybridOptimizer'],
    ['is_clipped', 'bool', 'true', 'logging / monitoring'],
    ['clip_reason', 'string', 'max_bid_exceeded', 'audit trail'],
]
output_table_n6 = Table(output_data_n6, colWidths=[1.4*inch, 1.2*inch, 1.3*inch, 1.6*inch])
output_table_n6.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(output_table_n6)
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("CONFIGURATION", subheading_style))
config_data_n6 = [
    ['Parameter', 'Value', 'Type', 'Source'],
    ['max_bid_multiplier', '2.0', 'float', 'config.yaml'],
    ['min_bid_multiplier', '0.5', 'float', 'config.yaml'],
    ['max_budget_multiplier', '1.5', 'float', 'config.yaml'],
    ['cooldown_period', '1800', 'int', 'config.yaml'],
]
config_table_n6 = Table(config_data_n6, colWidths=[1.5*inch, 1.2*inch, 1*inch, 1.6*inch])
config_table_n6.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(config_table_n6)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "<b>max_bid_multiplier:</b> AI can't increase bids more than 2x. Prevents runaway spending.",
    normal_style
))
story.append(Paragraph(
    "<b>min_bid_multiplier:</b> AI can't decrease bids more than 50%. Prevents under-bidding.",
    normal_style
))
story.append(Paragraph(
    "<b>cooldown_period:</b> Wait 30 minutes (1800 seconds) between major changes. Prevents over-optimization.",
    normal_style
))
story.append(PageBreak())

# NODE 7
story.append(Paragraph("Node 7: HybridOptimizer (Final Recommendations)", heading_style))
story.append(Paragraph("What It Does", subheading_style))
story.append(Paragraph(
    "This is the <b>decision hub</b> that combines DRL recommendations with LLM guidance and forecasting. Produces final recommendations that are: (1) optimized by AI, (2) reasonable per business logic, (3) forecasted to improve KPIs, and (4) explained in human-readable narratives.",
    normal_style
))

story.append(Paragraph("INPUTS", subheading_style))
input_data_n7 = [
    ['Field', 'Type', 'Sample', 'Source'],
    ['safe_action', 'float', '0.52', 'n6 SafeDRLAgent'],
    ['campaign_state', 'float32[42]', '[0.12, 0.04, ...]', 'n1 MockCampaignEnv'],
    ['campaign_id', 'string', 'cmp_7f3a', 'n1 MockCampaignEnv'],
]
input_table_n7 = Table(input_data_n7, colWidths=[1.4*inch, 1.2*inch, 1.3*inch, 1.6*inch])
input_table_n7.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(input_table_n7)
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("OUTPUTS", subheading_style))
output_data_n7 = [
    ['Field', 'Type', 'Sample', 'Used In'],
    ['recommendation', 'dict', '{bid_mult: 1.2, budget_+: $500}', 'campaign execution'],
    ['forecast', 'dict', '{roas: 2.5, cpa: $0.65}', 'monitoring / expectations'],
    ['narrative', 'string', 'Situation: High CTR, low CPA...', 'human explanation'],
    ['confidence', 'float', '0.87', 'decision confidence'],
]
output_table_n7 = Table(output_data_n7, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 1.8*inch])
output_table_n7.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(output_table_n7)
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("CONFIGURATION", subheading_style))
config_data_n7 = [
    ['Parameter', 'Value', 'Type', 'Source'],
    ['drl_weight', '0.4', 'float', 'config.yaml'],
    ['llm_weight', '0.3', 'float', 'config.yaml'],
    ['forecast_weight', '0.3', 'float', 'config.yaml'],
    ['narrative_detail', '5-part', 'string', 'config.yaml'],
]
config_table_n7 = Table(config_data_n7, colWidths=[1.5*inch, 1.2*inch, 1*inch, 1.6*inch])
config_table_n7.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3d6a8b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f0f0f0')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f0f0f0')])
]))
story.append(config_table_n7)
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "<b>drl_weight:</b> How much to trust DRL recommendations (40%).",
    normal_style
))
story.append(Paragraph(
    "<b>llm_weight:</b> How much to trust LLM guidance (30%).",
    normal_style
))
story.append(Paragraph(
    "<b>forecast_weight:</b> How much to trust outcome forecasts (30%).",
    normal_style
))
story.append(Paragraph(
    "<b>narrative_detail:</b> Format for explanation (5-part: Situation, Decision, Reasoning, Confidence, Reasonability).",
    normal_style
))

# Summary Page
story.append(PageBreak())
story.append(Paragraph("Quick Reference: Node Flow", heading_style))
story.append(Spacer(1, 0.2*inch))
story.append(Paragraph(
    "<b>Training Pipeline (Offline Learning):</b>",
    subheading_style
))
story.append(Paragraph(
    "n1 (Get Data) → n2 (Store) → n3 (Learn) → n4 (Save)",
    normal_style
))
story.append(Paragraph(
    "Node 1 fetches campaign data. Node 2 accumulates it. Node 3 trains on batches. Node 4 saves the learned model.",
    normal_style
))
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph(
    "<b>Inference Pipeline (Make Decisions):</b>",
    subheading_style
))
story.append(Paragraph(
    "n1 (Get Data) → n5 (Load Model) → n6 (Safety) → n7 (Recommend)",
    normal_style
))
story.append(Paragraph(
    "Node 1 gets current campaign state. Node 5 loads the trained model. Node 6 applies guardrails. Node 7 produces final recommendation.",
    normal_style
))
story.append(Spacer(1, 0.3*inch))

story.append(Paragraph("Key Concepts", heading_style))
story.append(Paragraph(
    "<b>state_vector:</b> 42-number snapshot of campaign (CTR, CVR, ROAS, CPA, constraints, etc.)",
    normal_style
))
story.append(Paragraph(
    "<b>action:</b> What to change (bid multiplier, budget adjustment, creative selection)",
    normal_style
))
story.append(Paragraph(
    "<b>reward:</b> How well the action worked (0-1 score)",
    normal_style
))
story.append(Paragraph(
    "<b>policy:</b> The trained neural network that maps states to actions",
    normal_style
))
story.append(Paragraph(
    "<b>constraint:</b> Business rule to enforce (max bid 2x, budget 1.5x, etc.)",
    normal_style
))
story.append(Paragraph(
    "<b>forecast:</b> Predicted impact of recommendation (expected ROAS, CPA, conversions)",
    normal_style
))

# Build PDF
doc.build(story)
print(f"✅ PDF created: {pdf_path}")
