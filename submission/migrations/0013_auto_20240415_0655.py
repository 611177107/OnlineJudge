# Generated by Django 3.2.9 on 2024-04-15 06:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('submission', '0012_auto_20180501_0436'),
    ]

    operations = [
        migrations.AddField(
            model_name='submission',
            name='gpt_message',
            field=models.TextField(null=True),
        ),
        migrations.AlterField(
            model_name='submission',
            name='info',
            field=models.JSONField(default=dict),
        ),
        migrations.AlterField(
            model_name='submission',
            name='statistic_info',
            field=models.JSONField(default=dict),
        ),
    ]
