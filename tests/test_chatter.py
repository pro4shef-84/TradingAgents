import unittest
from unittest.mock import patch
from tradingagents.pilot.chatter import news, discord, safe_url

NOW='2026-11-09T16:00:00Z'
def article(id=1, **kwargs):
    return dict(id=id,headline='Example',symbols=['CAVA'],created_at='2026-11-09T15:00:00Z',url='https://example.org/story',**kwargs)

class ChatterTests(unittest.TestCase):
    @patch('tradingagents.pilot.chatter.alpaca_get')
    def test_news_dedup_page_and_ticker_filter(self,get):
        get.side_effect=[{'news':[article(),dict(article(2),symbols=['AAPL'])],'next_page_token':'p'}, {'news':[article()], 'next_page_token':None}]
        items,status=news(['CAVA'],NOW)
        self.assertEqual(len(items),1);self.assertEqual(items[0]['kind'],'news')
        self.assertFalse(status['truncated']);self.assertEqual(get.call_args.args[1]['page_token'],'p')
    @patch('tradingagents.pilot.chatter.alpaca_get')
    def test_news_date_and_link_validation(self,get):
        get.return_value={'news':[dict(article(),created_at='2026-11-09T17:00:00Z'),dict(article(2),url='javascript:alert(1)'),dict(article(3),created_at='2025-01-01T00:00:00Z')]}
        items,_=news(['CAVA'],NOW)
        self.assertEqual(len(items),1);self.assertEqual(items[0]['url'],'')
        self.assertEqual(safe_url('https://user:password@example.org'),'')
    @patch('tradingagents.pilot.chatter.alpaca_get')
    def test_repeated_token_fails(self,get):
        get.return_value={'news':[], 'next_page_token':'repeat'}
        with self.assertRaises(ValueError):news(['CAVA'],NOW)
    @patch('tradingagents.pilot.chatter.alpaca_get')
    def test_news_cap_reported(self,get):
        get.side_effect=[{'news':[], 'next_page_token':str(i)} for i in range(10)]
        _,s=news(['CAVA'],NOW);self.assertTrue(s['truncated'])
    @patch.dict('os.environ',{},clear=True)
    @patch('tradingagents.pilot.chatter.dotenv_values',return_value={})
    def test_missing_discord_not_empty_connected(self,_):
        items,status=discord(['CAVA'],NOW)
        self.assertFalse(items);self.assertEqual(status['status'],'not_connected')
    @patch.dict('os.environ',{},clear=True)
    @patch('tradingagents.pilot.chatter.dotenv_values',return_value={'DISCORD_BOT_TOKEN':'test','DISCORD_CHANNEL_IDS':'123'})
    @patch('tradingagents.pilot.chatter.requests.get')
    def test_discord_explicit_symbols_and_permalink(self,get,_):
        from unittest.mock import Mock
        get.side_effect=[Mock(status_code=200,json=lambda:[{'id':'456','content':'Watching $CAVA, not plain AAPL','timestamp':NOW}]),Mock(status_code=200,json=lambda:{'guild_id':'789'})]
        items,status=discord(['CAVA','AAPL'],NOW)
        self.assertEqual(items[0]['symbols'],['CAVA'])
        self.assertEqual(items[0]['url'],'https://discord.com/channels/789/123/456')
        self.assertEqual(status['status'],'connected')


if __name__=='__main__':unittest.main()

class DiscordPaginationTests(unittest.TestCase):
    @patch.dict('os.environ',{},clear=True)
    @patch('tradingagents.pilot.chatter.dotenv_values',return_value={'DISCORD_BOT_TOKEN':'test','DISCORD_CHANNEL_IDS':'123'})
    @patch('tradingagents.pilot.chatter.requests.get')
    def test_pages_capped_with_before_cursor_and_edited_availability(self,get,_):
        from unittest.mock import Mock
        page=[{'id':str(500-i),'content':'$CAVA','timestamp':'2026-11-09T15:00:00Z'} for i in range(100)]
        second=[{'id':'300','content':'$CAVA','timestamp':'2026-11-09T15:00:00Z','edited_timestamp':'2026-11-09T17:00:00Z'}]
        get.side_effect=[Mock(status_code=200,json=lambda:page),Mock(status_code=200,json=lambda:second),Mock(status_code=200,json=lambda:{'guild_id':'789'})]
        items,status=discord(['CAVA'],NOW,max_pages=2)
        self.assertEqual(len(items),100)
        self.assertEqual(get.call_args_list[1].kwargs['params']['before'],'401')
        self.assertFalse(status['truncated'])
        self.assertEqual(status['message_count'],101)

    @patch.dict('os.environ',{},clear=True)
    @patch('tradingagents.pilot.chatter.dotenv_values',return_value={'DISCORD_BOT_TOKEN':'test','DISCORD_CHANNEL_IDS':'123'})
    @patch('tradingagents.pilot.chatter.requests.get')
    def test_repeated_message_cursor_rejected(self,get,_):
        from unittest.mock import Mock
        page=[{'id':str(500-i),'content':'$CAVA','timestamp':NOW} for i in range(100)]
        get.return_value=Mock(status_code=200,json=lambda:page)
        with self.assertRaisesRegex(ValueError,'cursor'):discord(['CAVA'],NOW,max_pages=3)
