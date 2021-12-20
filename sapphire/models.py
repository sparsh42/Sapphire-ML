from django.db import models
from django.contrib.auth.models import User
import random, os

def upload_csv(instance, filename):
    ext = filename.split('.')[-1]
    if instance.csv_name:
        finalname ='csv/{}'.format(instance.csv_name)
    else:
        rand_int = random.randint(100000000, 999999999)
        finalname ='csv/{}.{}'.format(rand_int, ext)
    return finalname
    
def upload_heatmap(instance, filename):
    ext = filename.split('.')[-1]
    if instance.csv_name:
        f_name = instance.csv_name.split('.')[0]
        finalname ='heatmap/{}.{}'.format(f_name, ext)
    else:
        return None
    return finalname

class CSV_model(models.Model):
    csv_id = models.AutoField(primary_key=True)
    csv_file = models.FileField(upload_to=upload_csv)
    heatmap = models.ImageField(upload_to=upload_heatmap, blank=True, null=True)
    csv_name = models.CharField(max_length = 100, blank=True, null=True)
    target_variable = models.CharField(max_length = 100, blank=True, null=True)
    train_test = models.IntegerField(blank=True, null=True)
    drop_column = models.BooleanField(default=False)
    outlier = models.BooleanField(default=False)

    def getfilename(self):
        return os.path.basename(self.csv_file.name)

    def delete(self, using=None, keep_parents=False):
        self.csv_file.storage.delete(self.csv_file.name)
        if self.heatmap:
            self.heatmap.storage.delete(self.heatmap.name)
        super().delete()

    def deleteCSVfile(self, using=None, keep_parents=False):
        self.csv_file.storage.delete(self.csv_file.name)
