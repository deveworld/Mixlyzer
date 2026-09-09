//! A small XML writer for attribute-only documents such as `DJ_PLAYLISTS`.
//!
//! Rekordbox collections carry all their data in attributes, so this writer
//! supports elements, attributes and nesting — and nothing else. That is
//! enough to emit the whole document and keeps the escaping rules small enough
//! to verify by reading them.
//!
//! # Why not `ElementTree`'s approach
//!
//! Python builds the tree with `xml.etree.ElementTree`, serialises it, and then
//! re-parses the bytes with `minidom` only to pretty-print them. Tag metadata
//! routinely contains control characters (a stray `\x00` or `\x01` left by a
//! tagger), which `ElementTree` happily writes and `minidom` then refuses to
//! read, raising `ExpatError` from inside the export and aborting the whole
//! library sync. Here the value is sanitised on the way in — characters XML 1.0
//! cannot represent are dropped — so no input can produce a document that will
//! not parse.

/// Whether `c` is a character XML 1.0 permits in a document.
///
/// Everything else — most notably the C0 control characters other than tab,
/// newline and carriage return — has no representation in XML 1.0 at all, not
/// even as a numeric character reference, so it can only be dropped.
pub fn is_xml_char(c: char) -> bool {
    matches!(c, '\u{9}' | '\u{A}' | '\u{D}')
        || ('\u{20}'..='\u{D7FF}').contains(&c)
        || ('\u{E000}'..='\u{FFFD}').contains(&c)
        || ('\u{10000}'..='\u{10FFFF}').contains(&c)
}

/// Drop every character XML 1.0 forbids, leaving the rest untouched.
///
/// Emoji and CJK text survive: they are ordinary characters above `\u{20}`.
pub fn sanitize_xml_text(value: &str) -> String {
    value.chars().filter(|c| is_xml_char(*c)).collect()
}

/// Sanitise and escape a string for use inside a double-quoted attribute.
///
/// `&`, `<`, `>`, `"` and `'` become entities. Tab, newline and carriage
/// return become numeric references: they are legal in an attribute value but
/// an XML parser normalises a literal one to a space, so writing them verbatim
/// would silently change the text on the round trip.
pub fn escape_attribute(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars().filter(|c| is_xml_char(*c)) {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\t' => out.push_str("&#9;"),
            '\n' => out.push_str("&#10;"),
            '\r' => out.push_str("&#13;"),
            other => out.push(other),
        }
    }
    out
}

/// An element: a name, ordered attributes, and child elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Element {
    name: String,
    attributes: Vec<(String, String)>,
    children: Vec<Element>,
}

impl Element {
    /// A childless, attribute-less element.
    ///
    /// Element and attribute names are written verbatim; every name in this
    /// crate is a literal from the Rekordbox schema, never user data.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Add an attribute, keeping insertion order. The value is escaped when
    /// the document is rendered, so callers pass raw text.
    #[must_use]
    pub fn attr(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.push((name.into(), value.into()));
        self
    }

    /// Append a child element.
    #[must_use]
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Append a child element in place, for building in a loop.
    pub fn push(&mut self, child: Element) {
        self.children.push(child);
    }

    /// The element's tag name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The raw (unescaped) value of an attribute, for inspection and tests.
    pub fn attribute(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// The element's children, in document order.
    pub fn children(&self) -> &[Element] {
        &self.children
    }

    /// Render this element as a complete document: XML declaration, then the
    /// tree pretty-printed with a two-space indent and a trailing newline.
    pub fn to_document(&self) -> String {
        let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        self.write(&mut out, 0);
        out
    }

    fn write(&self, out: &mut String, depth: usize) {
        for _ in 0..depth {
            out.push_str("  ");
        }
        out.push('<');
        out.push_str(&self.name);
        for (key, value) in &self.attributes {
            out.push(' ');
            out.push_str(key);
            out.push_str("=\"");
            out.push_str(&escape_attribute(value));
            out.push('"');
        }
        if self.children.is_empty() {
            out.push_str("/>\n");
            return;
        }
        out.push_str(">\n");
        for child in &self.children {
            child.write(out, depth + 1);
        }
        for _ in 0..depth {
            out.push_str("  ");
        }
        out.push_str("</");
        out.push_str(&self.name);
        out.push_str(">\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_document_starts_with_the_declaration_and_indents_by_two_spaces() {
        let doc = Element::new("ROOT")
            .attr("Version", "1.0.0")
            .child(Element::new("CHILD").child(Element::new("LEAF").attr("A", "1")))
            .to_document();
        assert_eq!(
            doc,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <ROOT Version=\"1.0.0\">\n\
             \x20 <CHILD>\n\
             \x20   <LEAF A=\"1\"/>\n\
             \x20 </CHILD>\n\
             </ROOT>\n"
        );
    }

    #[test]
    fn the_five_predefined_entities_are_escaped() {
        assert_eq!(
            escape_attribute(r#"Rock & Roll <"mix"> 'x'"#),
            "Rock &amp; Roll &lt;&quot;mix&quot;&gt; &apos;x&apos;"
        );
    }

    /// The bug that kills the Python export: a control character in a title
    /// makes `minidom` raise `ExpatError`. Here it is simply removed.
    #[test]
    fn control_characters_are_stripped_rather_than_aborting_the_export() {
        assert_eq!(escape_attribute("bad\u{0}\u{1}\u{1f}title"), "badtitle");
        assert_eq!(sanitize_xml_text("a\u{b}\u{c}b"), "ab");
        // Non-characters and the surrogate-adjacent block go too.
        assert_eq!(sanitize_xml_text("a\u{fffe}\u{ffff}b"), "ab");
    }

    #[test]
    fn whitespace_controls_survive_as_numeric_references() {
        assert_eq!(escape_attribute("a\tb\nc\rd"), "a&#9;b&#10;c&#13;d");
    }

    #[test]
    fn emoji_and_cjk_text_pass_through_unchanged() {
        let value = "🎧 ダンス 音楽 — 中文 café";
        assert_eq!(escape_attribute(value), value);
        assert_eq!(sanitize_xml_text(value), value);
    }

    #[test]
    fn hostile_metadata_still_produces_a_well_formed_attribute() {
        let doc = Element::new("TRACK")
            .attr("Name", "a\u{0}b\"c&d<e>f🎧")
            .to_document();
        assert!(doc.contains(r#"Name="ab&quot;c&amp;d&lt;e&gt;f🎧""#));
        assert!(!doc.contains('\u{0}'));
    }

    #[test]
    fn attribute_order_is_insertion_order() {
        let doc = Element::new("E").attr("Z", "1").attr("A", "2").to_document();
        assert!(doc.ends_with("<E Z=\"1\" A=\"2\"/>\n"));
    }

    #[test]
    fn attributes_can_be_read_back_raw() {
        let element = Element::new("E").attr("Name", "a&b");
        assert_eq!(element.attribute("Name"), Some("a&b"));
        assert_eq!(element.attribute("Missing"), None);
        assert_eq!(element.name(), "E");
    }
}
