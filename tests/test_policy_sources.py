import unittest

from src.data.policy_sources import body_text


class BodyTextTest(unittest.TestCase):
    def test_takes_only_the_marked_container(self):
        html = ('<div class="nav">导航</div><div id="UCAP-CONTENT"><p>第一段<span>内容</span></p>'
                '<script>var a = 1;</script><p>第二段</p><div><table><tr><td>指标</td><td>3.2</td></tr>'
                '</table></div></div><div class="foot">页脚</div>')
        self.assertEqual(body_text(html, ('UCAP-CONTENT',)), '第一段内容\n第二段\n\t指标\t3.2')

    def test_void_tags_do_not_end_the_container_early(self):
        html = '<span id="detailContent"><p>甲<br>乙<img src="a.png"></p><p>丙</p></span><p>外</p>'
        self.assertEqual(body_text(html, ('detailContent',)), '甲\n乙\n丙')

    def test_marker_must_be_a_whole_class_name(self):
        self.assertEqual(body_text('<div class="txt-content-wrap">x</div>', ('txt-content',)), '')

    def test_missing_container_gives_nothing(self):
        self.assertEqual(body_text('<div class="other">正文</div>', ('UCAP-CONTENT',)), '')


if __name__ == '__main__':
    unittest.main()
