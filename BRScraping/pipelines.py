# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class BrscrapingPipeline:
    def process_item(self, item, spider):
        return item

results = []
class CollectItemsPipeline:
    def process_item(self, item, spider):
        results.append(item)
        return item