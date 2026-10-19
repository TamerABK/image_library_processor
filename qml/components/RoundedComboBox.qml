pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

ComboBox {
    id: root

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }

    implicitHeight: metrics.toolbarButtonHeight
    leftPadding: metrics.spacingMedium
    rightPadding: metrics.spacingMedium + metrics.iconSizeSmall
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus

    contentItem: Text {
        text: root.displayText
        color: root.enabled ? theme.black : theme.disabledText
        font.family: typography.buttonFamily
        font.pixelSize: typography.button
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    indicator: Image {
        x: root.width - width - metrics.spacingSmall
        y: (root.height - height) / 2
        width: metrics.iconSizeSmall
        height: metrics.iconSizeSmall
        source: theme.iconUrl("chevron_down.svg")
        rotation: root.popup.visible ? 180 : 0
        fillMode: Image.PreserveAspectFit
    }

    background: Rectangle {
        radius: height / 2
        color: root.enabled ? theme.panelRaisedFill : theme.disabledFill
        border.width: root.activeFocus || root.popup.visible ? 2 : 1
        border.color: root.activeFocus || root.popup.visible ? theme.focusRing : theme.separator
    }

    delegate: ItemDelegate {
        id: option
        objectName: "roundedComboOption_" + index
        required property int index
        width: root.popup.availableWidth
        height: 40
        text: root.textAt(index)
        hoverEnabled: true
        highlighted: root.highlightedIndex === index
        leftPadding: metrics.spacingSmall
        rightPadding: metrics.spacingSmall
        contentItem: Text {
            text: option.text
            color: theme.black
            font.family: typography.bodyFamily
            font.pixelSize: typography.caption
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
        background: Rectangle {
            radius: metrics.radiusSmall
            color: option.highlighted || option.hovered ? theme.hoverFill : "transparent"
        }
        onClicked: {
            root.currentIndex = index
            root.popup.close()
        }
    }

    popup: Popup {
        y: root.height + metrics.spacingTiny
        width: root.width
        implicitHeight: Math.min(root.count * 40 + padding * 2, 248)
        padding: metrics.spacingTiny
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        contentItem: ListView {
            clip: true
            model: root.popup.visible ? root.delegateModel : null
            currentIndex: root.highlightedIndex
            boundsBehavior: Flickable.StopAtBounds
            ScrollIndicator.vertical: ScrollIndicator {}
        }
        background: Rectangle {
            radius: metrics.radiusMedium
            color: theme.panelRaisedFill
            border.width: 1
            border.color: theme.separator
        }
    }
}
