# Generated by Django 3.2.6 on 2021-08-22 05:34

from django.db import migrations, models
import sapphire.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CSV_model',
            fields=[
                ('csv_id', models.AutoField(primary_key=True, serialize=False)),
                ('csv_file', models.FileField(upload_to=sapphire.models.upload_csv)),
                ('target_variable', models.CharField(blank=True, max_length=100, null=True)),
                ('train_test', models.IntegerField(blank=True, null=True)),
            ],
        ),
    ]
